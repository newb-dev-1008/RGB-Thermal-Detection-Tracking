"""
Improved ONNX predictor with Kalman-filter-based multi-object tracker.

Improvements over predict_onnx_baseline.py:
1. Kalman filter for smooth track propagation between frames
2. Adaptive score threshold per modality (thermal: lower to compensate lower confidence)
3. Re-identification window: keeps "lost" tracks alive for N frames before pruning
4. Better NMS with class-agnostic suppression across detection queries
5. Modality auto-detection from streams_manifest.json for per-sequence thresholds

Usage::

    # Detection (same as baseline):
    python predict_onnx_improved.py detect \\
        --onnx exports/rfdetr/full/inference_model.onnx \\
        --dataset-dir ../dataset --split valid --out c2_preds_improved.json

    # MOT tracking with Kalman filter:
    python predict_onnx_improved.py mot \\
        --onnx exports/rfdetr/full/inference_model.onnx \\
        --eval-dir ../streams_eval --out-dir pred_mot_improved/

    # Evaluate:
    python evaluate_submissions.py detect \\
        --gt ../dataset/valid/_annotations.coco.json --preds c2_preds_improved.json
    python evaluate_submissions.py mot \\
        --gt-dir ../streams_eval --pred-dir pred_mot_improved/ --split valid
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _preprocess(im: Image.Image, size: int) -> np.ndarray:
    """Resize + ImageNet-normalize to NCHW float32."""
    rgb = im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    nchw = np.transpose(arr, (2, 0, 1))[None]
    return (nchw - IMAGENET_MEAN) / IMAGENET_STD


# ---------------------------------------------------------------------------
# Detection decoding
# ---------------------------------------------------------------------------

def _decode_rfdetr(
    dets: np.ndarray,
    labels: np.ndarray,
    w_px: int,
    h_px: int,
    num_classes: int,
    *,
    top_k: int = 100,
    score_thr: float = 0.0,
) -> list[tuple[int, float, list[float]]]:
    """Decode RF-DETR ONNX outputs to [(class_id, score, [x,y,w,h]) ...]."""
    probs = _sigmoid(labels[:, :num_classes])
    flat = probs.reshape(-1)
    k = min(top_k, len(flat))
    if k <= 0:
        return []
    top_idx = np.argpartition(-flat, k - 1)[:k]
    top_idx = top_idx[np.argsort(-flat[top_idx])]

    out = []
    for fidx in top_idx:
        score = float(flat[fidx])
        if score < score_thr:
            continue
        q = int(fidx // num_classes)
        cls = int(fidx % num_classes)
        cx, cy, bw, bh = dets[q]
        x1 = max(0.0, (float(cx) - float(bw) / 2) * w_px)
        y1 = max(0.0, (float(cy) - float(bh) / 2) * h_px)
        x2 = min(float(w_px), (float(cx) + float(bw) / 2) * w_px)
        y2 = min(float(h_px), (float(cy) + float(bh) / 2) * h_px)
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((cls, score, [x1, y1, x2 - x1, y2 - y1]))
    return out


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    ax1, ay1 = boxes_a[:, 0:1], boxes_a[:, 1:2]
    ax2, ay2 = ax1 + boxes_a[:, 2:3], ay1 + boxes_a[:, 3:4]
    bx1, by1 = boxes_b[:, 0][None, :], boxes_b[:, 1][None, :]
    bx2, by2 = bx1 + boxes_b[:, 2][None, :], by1 + boxes_b[:, 3][None, :]
    ix1, iy1 = np.maximum(ax1, bx1), np.maximum(ay1, by1)
    ix2, iy2 = np.minimum(ax2, bx2), np.minimum(ay2, by2)
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = boxes_a[:, 2:3] * boxes_a[:, 3:4]
    area_b = (boxes_b[:, 2] * boxes_b[:, 3])[None, :]
    union = np.maximum(area_a + area_b - inter, 1e-9)
    return inter / union


def _nms(
    dets: list[tuple[int, float, list[float]]],
    *,
    iou_thr: float = 0.5,
) -> list[tuple[int, float, list[float]]]:
    """Greedy class-agnostic NMS."""
    kept: list[tuple[int, float, list[float]]] = []
    for cand in sorted(dets, key=lambda x: -x[1]):
        cb = np.asarray([cand[2]])
        if any(
            float(_iou_matrix(cb, np.asarray([k[2]]))[0, 0]) >= iou_thr
            for k in kept
        ):
            continue
        kept.append(cand)
    return kept


# ---------------------------------------------------------------------------
# Kalman-filter tracker
# ---------------------------------------------------------------------------

class KalmanTrack:
    """
    Simple constant-velocity Kalman filter for one bounding box [x, y, w, h].

    State: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement: [cx, cy, w, h]
    """

    _F = None  # transition matrix (lazy init, same for all)
    _H = None  # measurement matrix
    _Q_base = None  # process noise
    _R_base = None  # measurement noise

    def __init__(self, bbox: list[float], track_id: int, class_id: int) -> None:
        self.id = track_id
        self.cls = class_id
        self.hits = 1
        self.misses = 0
        self.last_score = 0.0

        # Convert [x, y, w, h] → [cx, cy, w, h]
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2

        # State vector [cx, cy, w, h, vx, vy, vw, vh]
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(8, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 100.0  # high uncertainty on velocity

        if KalmanTrack._F is None:
            KalmanTrack._F = np.eye(8, dtype=np.float64)
            for i in range(4):
                KalmanTrack._F[i, i + 4] = 1.0
            KalmanTrack._H = np.zeros((4, 8), dtype=np.float64)
            for i in range(4):
                KalmanTrack._H[i, i] = 1.0
            KalmanTrack._Q_base = np.eye(8, dtype=np.float64)
            KalmanTrack._Q_base[4:, 4:] *= 0.01
            KalmanTrack._R_base = np.eye(4, dtype=np.float64) * 1.0

    def predict(self) -> np.ndarray:
        """Predict next state; return predicted measurement [cx, cy, w, h]."""
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self._Q_base
        return self._H @ self.x

    def update(self, bbox: list[float]) -> None:
        """Update with a matched detection."""
        x, y, w, h = bbox
        z = np.array([x + w / 2, y + h / 2, w, h], dtype=np.float64)
        innov = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + self._R_base
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.P = (np.eye(8) - K @ self._H) @ self.P
        self.hits += 1
        self.misses = 0

    def get_bbox(self) -> list[float]:
        """Return current state as [x, y, w, h]."""
        cx, cy, w, h = self.x[:4]
        w, h = max(1.0, float(w)), max(1.0, float(h))
        return [float(cx) - w / 2, float(cy) - h / 2, w, h]


class KalmanIOUTracker:
    """
    Multi-object tracker using Kalman filter + Hungarian IoU assignment.

    Args:
        iou_thr: IoU threshold for detection-track matching.
        max_age: Frames a track can be missing before removal (re-ID window).
        min_hits: Minimum hits before a track is confirmed (reduces FP).
        score_thr: Minimum detection score to initialise a new track.
    """

    def __init__(
        self,
        iou_thr: float = 0.3,
        max_age: int = 5,
        min_hits: int = 1,
        score_thr: float = 0.20,
    ) -> None:
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.min_hits = min_hits
        self.score_thr = score_thr
        self.tracks: dict[int, KalmanTrack] = {}
        self.next_id = 1

    def update(
        self, dets: list[tuple[int, float, list[float]]]
    ) -> list[tuple[int, int, float, list[float]]]:
        """
        Update tracker with current-frame detections.

        Args:
            dets: List of (class_id, score, [x, y, w, h]) from decoder.

        Returns:
            List of (track_id, class_id, score, [x, y, w, h]) for active tracks.
        """
        # Predict all existing tracks
        track_ids = list(self.tracks.keys())
        predicted_boxes = {}
        for tid in track_ids:
            pred_meas = self.tracks[tid].predict()
            cx, cy, w, h = pred_meas
            w, h = max(1.0, float(w)), max(1.0, float(h))
            predicted_boxes[tid] = [float(cx) - w / 2, float(cy) - h / 2, w, h]

        # Filter detections by score
        filtered_dets = [(c, s, b) for c, s, b in dets if s >= self.score_thr]

        # Match detections to tracks via Hungarian
        assigned_det = [-1] * len(filtered_dets)
        assigned_trk = set()
        if track_ids and filtered_dets:
            pred_arr = np.array([predicted_boxes[t] for t in track_ids])
            det_arr = np.array([d[2] for d in filtered_dets])
            iou = _iou_matrix(pred_arr, det_arr)  # (n_tracks, n_dets)
            cost = np.where(iou >= self.iou_thr, -iou, 1e6)
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if iou[r, c] >= self.iou_thr:
                    tid = track_ids[r]
                    self.tracks[tid].update(filtered_dets[c][2])
                    self.tracks[tid].last_score = filtered_dets[c][1]
                    assigned_det[c] = tid
                    assigned_trk.add(tid)

        # Increment miss counter for unmatched tracks
        for tid in track_ids:
            if tid not in assigned_trk:
                self.tracks[tid].misses += 1

        # Initialise new tracks for unmatched detections
        for i, (cls, score, bbox) in enumerate(filtered_dets):
            if assigned_det[i] < 0:
                new_tid = self.next_id
                self.next_id += 1
                trk = KalmanTrack(bbox, new_tid, cls)
                trk.last_score = score
                self.tracks[new_tid] = trk

        # Remove dead tracks
        dead = [tid for tid, t in self.tracks.items() if t.misses > self.max_age]
        for tid in dead:
            del self.tracks[tid]

        # Return active, confirmed tracks
        result = []
        for tid, trk in self.tracks.items():
            if trk.hits >= self.min_hits or trk.misses == 0:
                result.append((tid, trk.cls, trk.last_score, trk.get_bbox()))
        return result

    def reset(self) -> None:
        """Reset tracker state between sequences."""
        self.tracks.clear()
        self.next_id = 1


# ---------------------------------------------------------------------------
# Sequence modality lookup
# ---------------------------------------------------------------------------

def _load_modality_map(eval_dir: Path) -> dict[str, str]:
    """Return seq_name → modality from streams_manifest.json if present."""
    manifest_path = eval_dir / "streams_manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path) as f:
        doc = json.load(f)
    return {
        s["sequence"]: s.get("modality", "rgb")
        for s in doc.get("streams", [])
    }


# ---------------------------------------------------------------------------
# Detection prediction
# ---------------------------------------------------------------------------

def predict_detect(
    onnx_path: Path,
    dataset_dir: Path,
    split: str,
    out_json: Path,
    *,
    img_size: int = 384,
    score_thr: float = 0.0,
) -> dict[str, Any]:
    """Generate COCO predictions.json from an exported full RF-DETR ONNX model."""
    coco_path = dataset_dir / split / "_annotations.coco.json"
    gt_doc = json.loads(coco_path.read_text(encoding="utf-8"))
    num_classes = len(gt_doc["categories"])

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    preds = []
    latencies = []
    for im_rec in gt_doc["images"]:
        modality = im_rec.get("modality", "rgb")
        with Image.open(dataset_dir / split / im_rec["file_name"]) as im:
            inp = _preprocess(im, img_size)

        t0 = time.perf_counter()
        raw = sess.run(out_names, {in_name: inp})
        latencies.append((time.perf_counter() - t0) * 1000.0)

        outs = dict(zip(out_names, raw))
        for cid, score, bbox in _decode_rfdetr(
            outs["dets"][0],
            outs["labels"][0],
            int(im_rec["width"]),
            int(im_rec["height"]),
            num_classes,
            top_k=100,
            score_thr=score_thr,
        ):
            preds.append({
                "image_id": int(im_rec["id"]),
                "category_id": cid,
                "bbox": bbox,
                "score": score,
            })

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(preds), encoding="utf-8")
    return {
        "predictions": len(preds),
        "images": len(gt_doc["images"]),
        "median_ms": round(float(np.median(latencies)) if latencies else 0.0, 2),
    }


# ---------------------------------------------------------------------------
# MOT prediction with Kalman tracker
# ---------------------------------------------------------------------------

def predict_mot(
    onnx_path: Path,
    eval_dir: Path,
    out_dir: Path,
    *,
    num_classes: int = 3,
    img_size: int = 384,
    score_thr_rgb: float = 0.25,
    score_thr_thermal: float = 0.15,
    iou_thr: float = 0.30,
    max_age: int = 5,
) -> dict[str, Any]:
    """
    Run RF-DETR ONNX + Kalman IoU tracker on all MOT sequences.

    Uses per-modality score thresholds: thermal sequences use a lower threshold
    because thermal imagery typically produces lower confidence scores.
    """
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    out_dir.mkdir(parents=True, exist_ok=True)
    seqs = sorted(p for p in eval_dir.glob("seq_*") if (p / "video.mp4").exists())
    modality_map = _load_modality_map(eval_dir)
    total_boxes = 0

    with tempfile.TemporaryDirectory() as td:
        for seq_dir in seqs:
            seq_name = seq_dir.name
            modality = modality_map.get(seq_name, "rgb")
            score_thr = score_thr_thermal if modality == "thermal" else score_thr_rgb

            frames_dir = Path(td) / seq_name
            frames_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "ffmpeg", "-loglevel", "error",
                    "-i", str(seq_dir / "video.mp4"),
                    "-fps_mode", "passthrough",
                    str(frames_dir / "f_%06d.jpg"),
                ],
                check=True,
            )

            tracker = KalmanIOUTracker(
                iou_thr=iou_thr,
                max_age=max_age,
                min_hits=1,
                score_thr=score_thr,
            )
            lines: list[str] = []

            for f1, fpath in enumerate(sorted(frames_dir.glob("f_*.jpg")), 1):
                with Image.open(fpath) as im:
                    w_px, h_px = im.size
                    inp = _preprocess(im, img_size)

                outs = dict(zip(out_names, sess.run(out_names, {in_name: inp})))
                raw_dets = _decode_rfdetr(
                    outs["dets"][0],
                    outs["labels"][0],
                    w_px, h_px, num_classes,
                    top_k=20,
                    score_thr=max(0.0, score_thr - 0.05),  # slightly lower for tracker input
                )
                nms_dets = _nms(raw_dets, iou_thr=0.5)
                tracks = tracker.update(nms_dets)

                for tid, cls, score, box in tracks:
                    x, y, w, h = box
                    # Clamp to frame
                    x = max(0.0, x)
                    y = max(0.0, y)
                    w = min(w, w_px - x)
                    h = min(h, h_px - y)
                    if w <= 0 or h <= 0:
                        continue
                    lines.append(
                        f"{f1},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                        f"{score:.4f},{cls + 1},1.0"
                    )
                    total_boxes += 1

            (out_dir / f"{seq_name}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            print(f"  {seq_name} [{modality}] score_thr={score_thr} → {len(lines)} track rows")

    return {"sequences": len(seqs), "predicted_boxes": total_boxes}


# ---------------------------------------------------------------------------
# Argument parsing and main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="task", required=True)

    det = sub.add_parser("detect", help="COCO detection evaluation")
    det.add_argument("--onnx", type=Path, required=True)
    det.add_argument("--dataset-dir", type=Path, required=True)
    det.add_argument("--split", default="valid")
    det.add_argument("--out", type=Path, required=True)
    det.add_argument("--img-size", type=int, default=384)
    det.add_argument("--score-thr", type=float, default=0.0)

    mot = sub.add_parser("mot", help="MOT tracking evaluation")
    mot.add_argument("--onnx", type=Path, required=True)
    mot.add_argument("--eval-dir", type=Path, required=True)
    mot.add_argument("--out-dir", type=Path, required=True)
    mot.add_argument("--num-classes", type=int, default=3)
    mot.add_argument("--img-size", type=int, default=384)
    mot.add_argument(
        "--score-thr-rgb", type=float, default=0.25,
        help="Detection score threshold for RGB sequences."
    )
    mot.add_argument(
        "--score-thr-thermal", type=float, default=0.15,
        help="Detection score threshold for thermal sequences (lower = more sensitive)."
    )
    mot.add_argument(
        "--iou-thr", type=float, default=0.30,
        help="IoU threshold for Kalman track-detection matching."
    )
    mot.add_argument(
        "--max-age", type=int, default=5,
        help="Maximum frames a track can be absent (re-ID window)."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == "detect":
        summary = predict_detect(
            args.onnx,
            args.dataset_dir,
            args.split,
            args.out,
            img_size=args.img_size,
            score_thr=args.score_thr,
        )
    else:  # mot
        summary = predict_mot(
            args.onnx,
            args.eval_dir,
            args.out_dir,
            num_classes=args.num_classes,
            img_size=args.img_size,
            score_thr_rgb=args.score_thr_rgb,
            score_thr_thermal=args.score_thr_thermal,
            iou_thr=args.iou_thr,
            max_age=args.max_age,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
