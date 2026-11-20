"""Run exported static ONNX models via ONNX Runtime and emit evaluable submission files."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.optimize import linear_sum_assignment

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
SOUND_CLASSES = ("drone_buzz", "vehicle_mechanical", "explosion", "ambient_outdoor")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _preprocess_unit_rgb(im: Image.Image, size: int) -> np.ndarray:
    rgb = im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...]


def _preprocess_imagenet(im: Image.Image, size: int) -> np.ndarray:
    return (_preprocess_unit_rgb(im, size) - IMAGENET_MEAN) / IMAGENET_STD


def _decode_rfdetr_topk(
    dets: np.ndarray,
    labels: np.ndarray,
    w_px: int,
    h_px: int,
    num_classes: int,
    *,
    top_k: int = 100,
    score_thr: float = 0.0,
) -> list[tuple[int, float, list[float]]]:
    """Decode (Q, 4) cxcywh_norm and (Q, C+1) logits via top-K (query, class) scoring."""
    probs = _sigmoid(labels[:, :num_classes])
    flat = probs.reshape(-1)
    k = min(top_k, len(flat))
    if k <= 0:
        return []
    top_flat = np.argpartition(-flat, k - 1)[:k]
    top_flat = top_flat[np.argsort(-flat[top_flat])]

    out: list[tuple[int, float, list[float]]] = []
    for flat_idx in top_flat:
        score = float(flat[flat_idx])
        if score < score_thr:
            continue
        q_idx = int(flat_idx // num_classes)
        cls_id = int(flat_idx % num_classes)
        cx, cy, bw, bh = dets[q_idx]
        x1 = max(0.0, min(float(w_px - 1), (float(cx) - float(bw) / 2.0) * w_px))
        y1 = max(0.0, min(float(h_px - 1), (float(cy) - float(bh) / 2.0) * h_px))
        x2 = max(x1 + 1.0, min(float(w_px), (float(cx) + float(bw) / 2.0) * w_px))
        y2 = max(y1 + 1.0, min(float(h_px), (float(cy) + float(bh) / 2.0) * h_px))
        out.append((cls_id, score, [x1, y1, x2 - x1, y2 - y1]))
    return out


def _nms_dets(
    dets: list[tuple[int, float, list[float]]],
    *,
    iou_thr: float = 0.5,
) -> list[tuple[int, float, list[float]]]:
    """Suppress duplicate overlapping DETR detections greedily across queries."""
    kept: list[tuple[int, float, list[float]]] = []
    for cand in sorted(dets, key=lambda item: -item[1]):
        c_box = np.asarray([cand[2]], dtype=np.float64)
        if any(
            float(_iou_pairwise(c_box, np.asarray([k[2]], dtype=np.float64))[0, 0])
            >= iou_thr
            for k in kept
        ):
            continue
        kept.append(cand)
    return kept


def _iou_pairwise(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    ax1, ay1 = boxes_a[:, 0:1], boxes_a[:, 1:2]
    ax2, ay2 = ax1 + boxes_a[:, 2:3], ay1 + boxes_a[:, 3:4]
    bx1, by1 = boxes_b[:, 0][None, :], boxes_b[:, 1][None, :]
    bx2, by2 = bx1 + boxes_b[:, 2][None, :], by1 + boxes_b[:, 3][None, :]
    ix1, iy1 = np.maximum(ax1, bx1), np.maximum(ay1, by1)
    ix2, iy2 = np.minimum(ax2, bx2), np.minimum(ay2, by2)
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    union = np.maximum(
        boxes_a[:, 2:3] * boxes_a[:, 3:4]
        + (boxes_b[:, 2] * boxes_b[:, 3])[None, :]
        - inter,
        1e-9,
    )
    return inter / union


def predict_detect(
    onnx_path: Path,
    dataset_dir: Path,
    split: str,
    out_json: Path,
    *,
    img_size: int = 384,
) -> dict[str, Any]:
    """Generate COCO predictions.json from an exported full RF-DETR ONNX model."""
    coco_path = dataset_dir / split / "_annotations.coco.json"
    gt_doc = json.loads(coco_path.read_text(encoding="utf-8"))
    num_classes = len(gt_doc["categories"])
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    preds: list[dict[str, Any]] = []
    latencies: list[float] = []
    for im_rec in gt_doc["images"]:
        with Image.open(dataset_dir / split / im_rec["file_name"]) as im:
            inp = _preprocess_imagenet(im, img_size)
        t0 = time.perf_counter()
        raw = sess.run(out_names, {in_name: inp})
        latencies.append((time.perf_counter() - t0) * 1000.0)
        outs = dict(zip(out_names, raw, strict=True))
        for cid, score, bbox in _decode_rfdetr_topk(
            outs["dets"][0],
            outs["labels"][0],
            int(im_rec["width"]),
            int(im_rec["height"]),
            num_classes,
            top_k=100,
        ):
            preds.append(
                {
                    "image_id": int(im_rec["id"]),
                    "category_id": cid,
                    "bbox": bbox,
                    "score": score,
                }
            )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(preds), encoding="utf-8")
    return {
        "predictions": len(preds),
        "images": len(gt_doc["images"]),
        "median_ms": round(float(np.median(latencies)) if latencies else 0.0, 2),
    }


def predict_mot(
    onnx_path: Path,
    eval_dir: Path,
    out_dir: Path,
    *,
    num_classes: int = 3,
    img_size: int = 384,
    score_thr: float = 0.25,
) -> dict[str, Any]:
    """Run ONNX detector + greedy/Hungarian IoU tracker across MOT sequences."""
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    out_dir.mkdir(parents=True, exist_ok=True)
    seqs = sorted(p for p in eval_dir.glob("seq_*") if (p / "video.mp4").exists())
    total_boxes = 0

    with tempfile.TemporaryDirectory() as td:
        for seq_dir in seqs:
            frames_dir = Path(td) / seq_dir.name
            frames_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-i",
                    str(seq_dir / "video.mp4"),
                    "-fps_mode",
                    "passthrough",
                    str(frames_dir / "f_%06d.jpg"),
                ],
                check=True,
            )
            active: dict[int, np.ndarray] = {}
            next_tid = 1
            lines: list[str] = []
            for f1, fpath in enumerate(sorted(frames_dir.glob("f_*.jpg")), 1):
                with Image.open(fpath) as im:
                    w_px, h_px = im.size
                    inp = _preprocess_imagenet(im, img_size)
                outs = dict(
                    zip(out_names, sess.run(out_names, {in_name: inp}), strict=True)
                )
                dets = _nms_dets(
                    _decode_rfdetr_topk(
                        outs["dets"][0],
                        outs["labels"][0],
                        w_px,
                        h_px,
                        num_classes,
                        top_k=10,
                        score_thr=score_thr,
                    ),
                    iou_thr=0.5,
                )
                if not dets:
                    continue
                det_boxes = np.asarray([d[2] for d in dets], dtype=np.float64)
                assigned = [-1] * len(dets)
                if active:
                    tids = sorted(active)
                    prev = np.asarray([active[t] for t in tids], dtype=np.float64)
                    ious = _iou_pairwise(prev, det_boxes)
                    cost = np.where(ious >= 0.25, -ious, 1e6)
                    ri, ci = linear_sum_assignment(cost)
                    for r, c in zip(ri, ci, strict=True):
                        if ious[r, c] >= 0.25:
                            assigned[c] = tids[r]
                new_active: dict[int, np.ndarray] = {}
                for idx, (cid, score, box) in enumerate(dets):
                    tid = assigned[idx]
                    if tid < 0:
                        tid = next_tid
                        next_tid += 1
                    new_active[tid] = np.asarray(box, dtype=np.float64)
                    lines.append(
                        f"{f1},{tid},{box[0]:.2f},{box[1]:.2f},{box[2]:.2f},{box[3]:.2f},{score:.4f},{cid + 1},1.0"
                    )
                    total_boxes += 1
                active = new_active
            (out_dir / f"{seq_dir.name}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
    return {"sequences": len(seqs), "predicted_boxes": total_boxes}


def predict_triage(
    onnx_path: Path,
    dataset_dir: Path,
    split: str,
    out_jsonl: Path,
    *,
    img_size: int = 224,
) -> dict[str, Any]:
    """Generate Challenge 4 predictions JSONL from exported triage_distill.onnx."""
    rows_path = dataset_dir / split / "labels.jsonl"
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    preds: list[str] = []
    latencies: list[float] = []
    for row in rows:
        with Image.open(dataset_dir / row["image"]) as im:
            inp = _preprocess_unit_rgb(im, img_size)
        t0 = time.perf_counter()
        raw = sess.run(out_names, {in_name: inp})
        latencies.append((time.perf_counter() - t0) * 1000.0)
        outs = dict(zip(out_names, raw, strict=True))
        keep_prob = float(_sigmoid(outs["keep_logit"])[0])
        prio = max(0.0, min(1.0, float(outs["priority_score"][0])))
        preds.append(
            json.dumps(
                {
                    "id": row["id"],
                    "triage_keep": bool(keep_prob >= 0.5),
                    "priority_score": round(prio, 6),
                }
            )
        )
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text("\n".join(preds) + "\n", encoding="utf-8")
    return {
        "samples": len(preds),
        "median_ms": round(float(np.median(latencies)) if latencies else 0.0, 2),
    }


def resolve_wake_threshold(
    metadata: Mapping[str, str], override: float | None
) -> float:
    """Use an explicit override or the validation-tuned threshold stored in ONNX."""
    raw = override if override is not None else metadata.get("wake_threshold")
    if raw is None:
        raise ValueError(
            "ONNX model has no wake_threshold metadata. Re-export with the acoustic "
            "trainer or pass --wake-thresh using a threshold selected on valid/."
        )
    try:
        threshold = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Wake threshold must be a finite number in [0, 1]") from exc
    if isinstance(raw, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Wake threshold must be a finite number in [0, 1]")
    return threshold


def predict_acoustic(
    onnx_path: Path,
    challenge_dir: Path,
    split: str,
    out_jsonl: Path,
    *,
    img_size: int = 224,
    wake_thresh: float | None = None,
) -> dict[str, Any]:
    """Generate Challenge 5 predictions using the model's tuned threshold by default."""
    rows_path = challenge_dir / "dataset" / split / "labels.jsonl"
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    wake_thresh = resolve_wake_threshold(
        sess.get_modelmeta().custom_metadata_map, wake_thresh
    )
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    preds: list[str] = []
    latencies: list[float] = []
    for row in rows:
        with Image.open(challenge_dir / row["spectrogram"]) as im:
            inp = _preprocess_unit_rgb(im, img_size)
        t0 = time.perf_counter()
        raw = sess.run(out_names, {in_name: inp})
        latencies.append((time.perf_counter() - t0) * 1000.0)
        outs = dict(zip(out_names, raw, strict=True))
        wake_prob = float(_sigmoid(outs["wake_logit"])[0])
        cls_idx = int(np.argmax(outs["sound_class_logits"][0]))
        preds.append(
            json.dumps(
                {
                    "id": row["id"],
                    "wake_triggered": bool(wake_prob >= wake_thresh),
                    "sound_class": SOUND_CLASSES[cls_idx],
                }
            )
        )
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text("\n".join(preds) + "\n", encoding="utf-8")
    return {
        "samples": len(preds),
        "wake_thresh": wake_thresh,
        "median_ms": round(float(np.median(latencies)) if latencies else 0.0, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task", required=True)

    det = sub.add_parser("detect")
    det.add_argument("--onnx", type=Path, required=True)
    det.add_argument("--dataset-dir", type=Path, required=True)
    det.add_argument("--split", default="valid")
    det.add_argument("--out", type=Path, required=True)
    det.add_argument("--img-size", type=int, default=384)

    mot = sub.add_parser("mot")
    mot.add_argument("--onnx", type=Path, required=True)
    mot.add_argument("--eval-dir", type=Path, required=True)
    mot.add_argument("--out-dir", type=Path, required=True)
    mot.add_argument("--num-classes", type=int, default=3)
    mot.add_argument("--img-size", type=int, default=384)
    mot.add_argument("--score-thr", type=float, default=0.25)

    tri = sub.add_parser("triage")
    tri.add_argument("--onnx", type=Path, required=True)
    tri.add_argument("--dataset-dir", type=Path, required=True)
    tri.add_argument("--split", default="valid")
    tri.add_argument("--out", type=Path, required=True)
    tri.add_argument("--img-size", type=int, default=224)

    ac = sub.add_parser("acoustic")
    ac.add_argument("--onnx", type=Path, required=True)
    ac.add_argument("--challenge-dir", type=Path, required=True)
    ac.add_argument("--split", default="valid")
    ac.add_argument("--out", type=Path, required=True)
    ac.add_argument("--img-size", type=int, default=224)
    ac.add_argument(
        "--wake-thresh",
        type=float,
        default=None,
        help="Override the tuned ONNX wake_threshold; required for older models without metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == "detect":
        summary = predict_detect(
            args.onnx, args.dataset_dir, args.split, args.out, img_size=args.img_size
        )
    elif args.task == "mot":
        summary = predict_mot(
            args.onnx,
            args.eval_dir,
            args.out_dir,
            num_classes=args.num_classes,
            img_size=args.img_size,
            score_thr=args.score_thr,
        )
    elif args.task == "triage":
        summary = predict_triage(
            args.onnx, args.dataset_dir, args.split, args.out, img_size=args.img_size
        )
    else:
        summary = predict_acoustic(
            args.onnx,
            args.challenge_dir,
            args.split,
            args.out,
            img_size=args.img_size,
            wake_thresh=args.wake_thresh,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
