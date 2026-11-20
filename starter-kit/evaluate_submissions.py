"""Official hackathon scorer for Challenges 1, 2, 4, and 5."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

SOUND_CLASSES: frozenset[str] = frozenset(
    {"drone_buzz", "vehicle_mechanical", "explosion", "ambient_outdoor"}
)


def _reject_non_finite_json(token: str) -> Any:
    raise ValueError(f"Non-finite JSON numeric constant is not allowed: {token!r}")


def _loads_strict_json(raw: str, source: Path) -> Any:
    """Parse JSON while rejecting ``NaN``, ``Infinity``, and ``-Infinity``."""
    try:
        return json.loads(raw, parse_constant=_reject_non_finite_json)
    except ValueError as exc:
        raise ValueError(f"Invalid JSON in {source}: {exc}") from exc


def _as_strict_int(value: Any, *, field: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected integer for {field!r} in {source}, got {value!r}")
    return value


def _as_finite_float(value: Any, *, field: str, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Expected finite number for {field!r} in {source}, got {value!r}"
        )
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"Non-finite value for {field!r} in {source}: {value!r}")
    return out


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between ``[x, y, w, h]`` arrays."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    ax1, ay1 = boxes_a[:, 0:1], boxes_a[:, 1:2]
    ax2, ay2 = ax1 + boxes_a[:, 2:3], ay1 + boxes_a[:, 3:4]
    bx1, by1 = boxes_b[:, 0][None, :], boxes_b[:, 1][None, :]
    bx2, by2 = bx1 + boxes_b[:, 2][None, :], by1 + boxes_b[:, 3][None, :]

    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = np.maximum(0.0, boxes_a[:, 2:3]) * np.maximum(0.0, boxes_a[:, 3:4])
    area_b = (np.maximum(0.0, boxes_b[:, 2]) * np.maximum(0.0, boxes_b[:, 3]))[None, :]
    union = np.maximum(area_a + area_b - inter, 1e-9)
    return inter / union


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute 101-point COCO interpolated Average Precision."""
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    recall_points = np.linspace(0.0, 1.0, 101)
    ap = np.mean(
        [mpre[mrec >= t].max() if np.any(mrec >= t) else 0.0 for t in recall_points]
    )
    return float(ap)


def _validate_coco_predictions(
    raw_preds: Any,
    preds_json: Path,
    *,
    max_dets_per_image: int,
) -> dict[int, list[dict[str, Any]]]:
    """Validate prediction schema, finiteness, positive dims, and top-k image cap."""
    if not isinstance(raw_preds, list):
        raise ValueError(f"Expected JSON list of predictions in {preds_json}")

    by_image: dict[int, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for idx, pred in enumerate(raw_preds):
        ctx = f"{preds_json}[{idx}]"
        if not isinstance(pred, dict):
            raise ValueError(f"Prediction at {ctx} must be an object")
        image_id = _as_strict_int(pred.get("image_id"), field="image_id", source=ctx)
        cat_id = _as_strict_int(
            pred.get("category_id"), field="category_id", source=ctx
        )
        score = _as_finite_float(pred.get("score"), field="score", source=ctx)
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"Prediction score out of [0, 1] at {ctx}: {score}")

        raw_bbox = pred.get("bbox")
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            raise ValueError(f"Expected 4-element bbox at {ctx}, got {raw_bbox!r}")
        bbox = [
            _as_finite_float(raw_bbox[i], field=f"bbox[{i}]", source=ctx)
            for i in range(4)
        ]
        if bbox[2] <= 0.0 or bbox[3] <= 0.0:
            raise ValueError(f"Bbox width/height must be > 0 at {ctx}: {bbox}")

        clean_pred = {
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": bbox,
            "score": score,
        }
        by_image[image_id].append((score, clean_pred))

    preds_by_cat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for items in by_image.values():
        items.sort(key=lambda pair: -pair[0])
        for _, clean_pred in items[:max_dets_per_image]:
            preds_by_cat[int(clean_pred["category_id"])].append(clean_pred)
    return preds_by_cat


def evaluate_coco_detection(
    gt_json: Path,
    preds_json: Path,
    *,
    small_area_thresh: float = 32.0 * 32.0,
    max_dets: int = 100,
) -> dict[str, Any]:
    """Compute COCO mAP50, mAP50:95, Small-Object AP, and per-modality AP50."""
    gt_doc = _loads_strict_json(gt_json.read_text(encoding="utf-8"), gt_json)
    raw_preds = _loads_strict_json(preds_json.read_text(encoding="utf-8"), preds_json)

    images_by_id = {int(im["id"]): im for im in gt_doc["images"]}
    if not images_by_id:
        raise ValueError(f"Empty COCO images list in {gt_json}")
    if not gt_doc.get("annotations"):
        raise ValueError(
            f"Ground truth {gt_json} contains no ground-truth annotations (cannot score stripped public test split)"
        )
    cat_ids = sorted(int(c["id"]) for c in gt_doc["categories"])
    iou_thresholds = np.linspace(0.50, 0.95, 10)

    gt_by_cat: dict[int, dict[int, list[tuple[list[float], float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for ann in gt_doc["annotations"]:
        area = float(ann.get("area", ann["bbox"][2] * ann["bbox"][3]))
        gt_by_cat[int(ann["category_id"])][int(ann["image_id"])].append(
            (ann["bbox"], area)
        )

    preds_by_cat = _validate_coco_predictions(
        raw_preds, preds_json, max_dets_per_image=max_dets
    )

    def _eval_subset(
        allowed_image_ids: set[int], area_max: float | None = None
    ) -> tuple[float, float]:
        aps_50: list[float] = []
        aps_all: list[float] = []
        for cid in cat_ids:
            gt_boxes: dict[int, np.ndarray] = {}
            gt_ignore: dict[int, np.ndarray] = {}
            total_gt = 0
            for iid, entries in gt_by_cat.get(cid, {}).items():
                if iid not in allowed_image_ids:
                    continue
                gt_boxes[iid] = np.asarray([e[0] for e in entries], dtype=np.float64)
                ign = np.asarray(
                    [
                        False if area_max is None else (e[1] >= area_max)
                        for e in entries
                    ],
                    dtype=bool,
                )
                gt_ignore[iid] = ign
                total_gt += int((~ign).sum())
            if total_gt == 0:
                continue

            cls_preds = [
                p
                for p in preds_by_cat.get(cid, [])
                if int(p["image_id"]) in allowed_image_ids
            ]
            cls_preds.sort(key=lambda p: -float(p["score"]))
            if not cls_preds:
                aps_50.append(0.0)
                aps_all.append(0.0)
                continue

            thr_aps: list[float] = []
            for thr in iou_thresholds:
                matched = {
                    iid: np.zeros(len(arr), dtype=bool) for iid, arr in gt_boxes.items()
                }
                tp_list: list[float] = []
                fp_list: list[float] = []
                for pred in cls_preds:
                    iid = int(pred["image_id"])
                    pbox = pred["bbox"]
                    pred_area = float(pbox[2] * pbox[3])
                    if iid not in gt_boxes:
                        if area_max is None or pred_area < area_max:
                            tp_list.append(0.0)
                            fp_list.append(1.0)
                        continue
                    ious = _iou_matrix(
                        np.asarray([pbox], dtype=np.float64), gt_boxes[iid]
                    )[0]
                    order = np.argsort(-ious)
                    chosen_j = -1
                    for j in order:
                        if not (ious[j] >= thr):
                            break
                        if matched[iid][j]:
                            continue
                        if not gt_ignore[iid][j]:
                            chosen_j = int(j)
                            break
                        if chosen_j < 0:
                            chosen_j = int(j)
                    if chosen_j >= 0:
                        matched[iid][chosen_j] = True
                        if not gt_ignore[iid][chosen_j]:
                            tp_list.append(1.0)
                            fp_list.append(0.0)
                    elif area_max is None or pred_area < area_max:
                        tp_list.append(0.0)
                        fp_list.append(1.0)
                if not tp_list:
                    thr_aps.append(0.0)
                    continue
                cum_tp = np.cumsum(np.asarray(tp_list, dtype=np.float64))
                cum_fp = np.cumsum(np.asarray(fp_list, dtype=np.float64))
                rec = cum_tp / max(1, total_gt)
                prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
                thr_aps.append(_compute_ap(rec, prec))
            aps_50.append(thr_aps[0])
            aps_all.append(float(np.mean(thr_aps)))
        return (
            float(np.mean(aps_50)) if aps_50 else 0.0,
            float(np.mean(aps_all)) if aps_all else 0.0,
        )

    all_ids = set(images_by_id)
    map50, map50_95 = _eval_subset(all_ids)
    small_ap50, small_ap = _eval_subset(all_ids, area_max=small_area_thresh)
    result: dict[str, Any] = {
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "small_AP50": round(small_ap50, 4),
        "small_AP50_95": round(small_ap, 4),
        "num_images": len(all_ids),
    }

    modalities = {
        im.get("modality") for im in images_by_id.values() if im.get("modality")
    }
    for mod in sorted(modalities):
        mod_ids = {iid for iid, im in images_by_id.items() if im.get("modality") == mod}
        m50, m_all = _eval_subset(mod_ids)
        result[f"{mod}_mAP50"] = round(m50, 4)
        result[f"{mod}_mAP50_95"] = round(m_all, 4)
    return result


def _load_mot_txt(
    path: Path, *, max_frame: int | None = None
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Parse and validate 1-based MOTChallenge CSV lines."""
    frames: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    if not path.exists():
        return frames
    seen_ft: set[tuple[int, int]] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 6:
            raise ValueError(f"Malformed MOT row in {path}:{lineno}: {line!r}")
        try:
            raw_f = float(parts[0])
            raw_t = float(parts[1])
            coords = [
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
            ]
        except (ValueError, OverflowError) as exc:
            raise ValueError(
                f"Non-numeric MOT row in {path}:{lineno}: {line!r}"
            ) from exc

        if not (
            math.isfinite(raw_f)
            and math.isfinite(raw_t)
            and raw_f.is_integer()
            and raw_t.is_integer()
        ):
            raise ValueError(
                f"Frame/track IDs must be finite integers in {path}:{lineno}: {line!r}"
            )
        if not all(math.isfinite(v) for v in coords):
            raise ValueError(f"Non-finite MOT bbox in {path}:{lineno}: {line!r}")
        frame_id = int(raw_f)
        track_id = int(raw_t)
        if frame_id < 1:
            raise ValueError(
                f"MOT frame numbers must be 1-based (>= 1) in {path}:{lineno}: {frame_id}"
            )
        if max_frame is not None and frame_id > max_frame:
            raise ValueError(
                f"MOT frame {frame_id} exceeds sequence frame_count={max_frame} in {path}:{lineno}"
            )
        if track_id < 0:
            raise ValueError(f"Negative MOT track_id in {path}:{lineno}: {track_id}")
        if coords[2] <= 0.0 or coords[3] <= 0.0:
            raise ValueError(f"MOT bbox width/height must be > 0 in {path}:{lineno}")
        ft = (frame_id, track_id)
        if ft in seen_ft:
            raise ValueError(
                f"Duplicate (frame={frame_id}, track_id={track_id}) in {path}:{lineno}"
            )
        seen_ft.add(ft)
        frames[frame_id].append((track_id, np.asarray(coords, dtype=np.float64)))
    return frames


def _load_eval_manifest(gt_dir: Path) -> dict[str, dict[str, Any]]:
    for name in ("sequences_manifest.json", "streams_manifest.json"):
        mpath = gt_dir / name
        if mpath.exists():
            doc = _loads_strict_json(mpath.read_text(encoding="utf-8"), mpath)
            entries = doc.get("sequences") or doc.get("streams") or []
            return {str(e["sequence"]): e for e in entries if "sequence" in e}
    return {}


def evaluate_mot_tracking(
    gt_dir: Path,
    pred_dir: Path,
    iou_thresh: float = 0.5,
    *,
    split: str = "all",
) -> dict[str, float]:
    """Compute class-agnostic MOTA, per-frame-paired IDF1, and Track Continuity."""
    all_gt_files = sorted(gt_dir.glob("seq_*/gt/gt.txt"))
    if not all_gt_files:
        raise FileNotFoundError(
            f"No MOT ground-truth files found under {gt_dir}/seq_*/gt/gt.txt"
        )

    manifest = _load_eval_manifest(gt_dir)
    if manifest:
        expected_seqs = [
            seq
            for seq, entry in sorted(manifest.items())
            if split == "all" or entry.get("split") == split
        ]
        missing_gt = [
            seq
            for seq in expected_seqs
            if not (gt_dir / seq / "gt" / "gt.txt").is_file()
        ]
        if missing_gt:
            raise ValueError(
                f"Missing MOT ground-truth gt.txt for split={split!r}: {missing_gt[:5]} (cannot score stripped public test split)"
            )
    if split != "all":
        gt_files = [
            f
            for f in all_gt_files
            if manifest.get(f.parents[1].name, {}).get("split") == split
        ]
        if not gt_files:
            raise ValueError(f"No MOT sequences found for split={split!r} in {gt_dir}")
    else:
        gt_files = all_gt_files

    known_seqs = (
        set(manifest) if manifest else {f.parents[1].name for f in all_gt_files}
    )
    for pred_txt in sorted(pred_dir.glob("seq_*.txt")):
        if pred_txt.stem not in known_seqs:
            raise ValueError(
                f"Unknown prediction sequence file {pred_txt.name} not present in {gt_dir}"
            )

    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    id_switches = 0
    total_pred_count = 0
    gt_total_frames: dict[tuple[str, int], int] = defaultdict(int)
    gt_covered_frames: dict[tuple[str, int], int] = defaultdict(int)
    id_pair_by_seq: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    gt_ids_by_seq: dict[str, list[int]] = defaultdict(list)
    pred_ids_by_seq: dict[str, list[int]] = defaultdict(list)

    for gt_file in gt_files:
        seq_name = gt_file.parents[1].name
        max_frame_raw = manifest.get(seq_name, {}).get("frame_count")
        max_frame = int(max_frame_raw) if max_frame_raw is not None else None
        pred_file = pred_dir / f"{seq_name}.txt"
        gt_frames = _load_mot_txt(gt_file, max_frame=max_frame)
        pred_frames = _load_mot_txt(pred_file, max_frame=max_frame)
        total_pred_count += sum(len(items) for items in pred_frames.values())
        last_match: dict[int, int] = {}
        seq_gt_ids: set[int] = set()
        seq_pred_ids: set[int] = set()
        seq_pairs: dict[tuple[int, int], int] = defaultdict(int)

        for frame_id in sorted(set(gt_frames) | set(pred_frames)):
            g_list = gt_frames.get(frame_id, [])
            p_list = pred_frames.get(frame_id, [])
            total_gt += len(g_list)
            for gid, _ in g_list:
                gt_total_frames[(seq_name, gid)] += 1
                seq_gt_ids.add(gid)
            for pid, _ in p_list:
                seq_pred_ids.add(pid)

            if not g_list or not p_list:
                total_fn += len(g_list)
                total_fp += len(p_list)
                continue

            g_boxes = np.stack([b for _, b in g_list])
            p_boxes = np.stack([b for _, b in p_list])
            ious = _iou_matrix(g_boxes, p_boxes)
            cost = np.where(ious >= iou_thresh, -ious, 1e6)
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_g, matched_p = set(), set()
            for r, c in zip(row_ind, col_ind, strict=True):
                if not (ious[r, c] >= iou_thresh):
                    continue
                matched_g.add(r)
                matched_p.add(c)
                gid = g_list[r][0]
                pid = p_list[c][0]
                total_tp += 1
                gt_covered_frames[(seq_name, gid)] += 1
                seq_pairs[(gid, pid)] += 1
                if gid in last_match and last_match[gid] != pid:
                    id_switches += 1
                last_match[gid] = pid
            total_fn += len(g_list) - len(matched_g)
            total_fp += len(p_list) - len(matched_p)

        gt_ids_by_seq[seq_name] = sorted(seq_gt_ids)
        pred_ids_by_seq[seq_name] = sorted(seq_pred_ids)
        for (gid, pid), cnt in seq_pairs.items():
            id_pair_by_seq[seq_name].append((gid, pid, cnt))

    mota = 1.0 - (total_fn + total_fp + id_switches) / max(1, total_gt)
    total_gt_count = sum(gt_total_frames.values())
    idtp = 0.0
    for seq_name, g_ids in gt_ids_by_seq.items():
        p_ids = pred_ids_by_seq.get(seq_name, [])
        if not g_ids or not p_ids:
            continue
        g_idx = {gid: i for i, gid in enumerate(g_ids)}
        p_idx = {pid: j for j, pid in enumerate(p_ids)}
        mat = np.zeros((len(g_ids), len(p_ids)), dtype=np.float64)
        for gid, pid, cnt in id_pair_by_seq[seq_name]:
            mat[g_idx[gid], p_idx[pid]] = cnt
        ri, ci = linear_sum_assignment(-mat)
        idtp += float(mat[ri, ci].sum())
    idf1 = (2.0 * idtp) / max(1.0, float(total_gt_count + total_pred_count))

    continuity = float(
        np.mean([gt_covered_frames[k] / max(1, v) for k, v in gt_total_frames.items()])
        if gt_total_frames
        else 0.0
    )
    return {
        "MOTA": round(float(mota), 4),
        "IDF1": round(float(idf1), 4),
        "track_continuity": round(continuity, 4),
        "id_switches": id_switches,
        "total_gt_boxes": total_gt,
    }


def _load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    """Parse JSONL strictly, rejecting non-finite constants and duplicate IDs."""
    records: dict[str, dict[str, Any]] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = _loads_strict_json(line, path)
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object at {path}:{lineno}")
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"Missing or non-string 'id' at {path}:{lineno}")
        if item_id in records:
            raise ValueError(f"Duplicate id {item_id!r} at {path}:{lineno}")
        records[item_id] = row
    return records


def _validate_exact_ids(
    gt_ids: set[str], pred_ids: set[str], gt_path: Path, pred_path: Path
) -> list[str]:
    missing = sorted(gt_ids - pred_ids)
    extra = sorted(pred_ids - gt_ids)
    if missing:
        raise ValueError(
            f"Missing {len(missing)} predictions in {pred_path} vs {gt_path} "
            f"(first missing: {missing[:3]})"
        )
    if extra:
        raise ValueError(
            f"Found {len(extra)} unexpected IDs in {pred_path} not in {gt_path} "
            f"(first extra: {extra[:3]})"
        )
    return sorted(gt_ids)


def evaluate_edge_triage(gt_jsonl: Path, pred_jsonl: Path) -> dict[str, float]:
    """Score Challenge 4 Edge Triage Distillation predictions (with pessimistic tie-breaking)."""
    gt_rows = _load_jsonl_by_id(gt_jsonl)
    pred_rows = _load_jsonl_by_id(pred_jsonl)
    if not gt_rows:
        raise ValueError(f"Empty ground truth in {gt_jsonl}")
    ids = _validate_exact_ids(set(gt_rows), set(pred_rows), gt_jsonl, pred_jsonl)

    score_list: list[float] = []
    for i in ids:
        row = pred_rows[i]
        score = _as_finite_float(
            row.get("priority_score"),
            field="priority_score",
            source=f"{pred_jsonl}:{i}",
        )
        if "triage_keep" in row and not isinstance(row["triage_keep"], bool):
            raise ValueError(
                f"Expected boolean 'triage_keep' for id={i!r} in {pred_jsonl}"
            )
        score_list.append(score)

    scores = np.asarray(score_list, dtype=np.float64)
    for i in ids:
        gt_row = gt_rows[i]
        if (
            not isinstance(gt_row.get("triage_keep"), bool)
            or "priority_score" not in gt_row
        ):
            raise ValueError(
                f"Ground truth {gt_jsonl} is missing ground-truth labels ('triage_keep'/'priority_score') for id={i!r}"
            )
    y_keep = np.asarray([bool(gt_rows[i]["triage_keep"]) for i in ids], dtype=bool)
    y_prio = np.asarray(
        [
            _as_finite_float(
                gt_rows[i]["priority_score"],
                field="priority_score",
                source=f"{gt_jsonl}:{i}",
            )
            for i in ids
        ],
        dtype=np.float64,
    )
    y_high = y_keep & (y_prio >= 0.7)

    # Pessimistic tie-breaking: among equal predicted scores, place y_high==False first
    # so constant/degenerate predictions never gain free positives from ID ordering.
    order = np.lexsort((np.arange(len(ids)), y_high.astype(np.int32), -scores))

    results: dict[str, float] = {}
    positives = int(y_high.sum())
    for label, frac in (("recall_at_1pct", 0.01), ("recall_at_5pct", 0.05)):
        k = max(1, int(round(len(ids) * frac)))
        top = order[:k]
        results[label] = round(float(y_high[top].sum() / max(1, min(k, positives))), 4)

    if np.ptp(scores) == 0.0 or np.ptp(y_prio) == 0.0:
        rho = 0.0
    else:
        raw_rho = float(spearmanr(scores, y_prio).statistic)
        rho = 0.0 if not math.isfinite(raw_rho) else raw_rho
    results["priority_spearman"] = round(rho, 4)
    return results


def evaluate_acoustic_cuas(gt_jsonl: Path, pred_jsonl: Path) -> dict[str, float]:
    """Score Challenge 5 Acoustic Wake -> Optical Slew C-UAS predictions."""
    gt_rows = _load_jsonl_by_id(gt_jsonl)
    pred_rows = _load_jsonl_by_id(pred_jsonl)
    if not gt_rows:
        raise ValueError(f"Empty ground truth in {gt_jsonl}")
    ids = _validate_exact_ids(set(gt_rows), set(pred_rows), gt_jsonl, pred_jsonl)

    pred_wakes: list[bool] = []
    pred_classes: list[str] = []
    for i in ids:
        row = pred_rows[i]
        wake_val = row.get("wake_triggered")
        cls_val = row.get("sound_class")
        if not isinstance(wake_val, bool):
            raise ValueError(
                f"Expected boolean 'wake_triggered' for id={i!r} in {pred_jsonl}, got {wake_val!r}"
            )
        if not isinstance(cls_val, str) or not cls_val:
            raise ValueError(
                f"Expected string 'sound_class' for id={i!r} in {pred_jsonl}, got {cls_val!r}"
            )
        if cls_val not in SOUND_CLASSES:
            raise ValueError(
                f"Unknown sound_class {cls_val!r} for id={i!r} in {pred_jsonl}; expected one of {sorted(SOUND_CLASSES)}"
            )
        pred_wakes.append(wake_val)
        pred_classes.append(cls_val)

    for i in ids:
        gt_row = gt_rows[i]
        if (
            not isinstance(gt_row.get("has_drone_audio"), bool)
            or gt_row.get("sound_class") not in SOUND_CLASSES
        ):
            raise ValueError(
                f"Ground truth {gt_jsonl} is missing ground-truth labels ('has_drone_audio'/'sound_class') for id={i!r}"
            )
    y_wake = np.asarray([bool(gt_rows[i]["has_drone_audio"]) for i in ids], dtype=bool)
    p_wake = np.asarray(pred_wakes, dtype=bool)
    tp = int((p_wake & y_wake).sum())
    fp = int((p_wake & ~y_wake).sum())
    fn = int((~p_wake & y_wake).sum())
    wake_f1 = (2.0 * tp) / max(1, 2 * tp + fp + fn)
    duty_cycle = float(p_wake.mean())
    cls_acc = float(
        np.mean(
            [
                pred_classes[idx] == gt_rows[i]["sound_class"]
                for idx, i in enumerate(ids)
            ]
        )
    )
    return {
        "wake_F1": round(wake_f1, 4),
        "sound_class_accuracy": round(cls_acc, 4),
        "optical_duty_cycle": round(duty_cycle, 4),
        "energy_savings_pct": round((1.0 - duty_cycle) * 100.0, 2),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task", required=True)
    det = sub.add_parser("detect")
    det.add_argument("--gt", type=Path, required=True)
    det.add_argument("--preds", type=Path, required=True)
    mot = sub.add_parser("mot")
    mot.add_argument("--gt-dir", type=Path, required=True)
    mot.add_argument("--pred-dir", type=Path, required=True)
    mot.add_argument("--split", choices=("all", "valid", "test"), default="valid")
    tri = sub.add_parser("triage")
    tri.add_argument("--gt", type=Path, required=True)
    tri.add_argument("--preds", type=Path, required=True)
    ac = sub.add_parser("acoustic")
    ac.add_argument("--gt", type=Path, required=True)
    ac.add_argument("--preds", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.task == "detect":
        out = evaluate_coco_detection(args.gt, args.preds)
    elif args.task == "mot":
        out = evaluate_mot_tracking(args.gt_dir, args.pred_dir, split=args.split)
    elif args.task == "triage":
        out = evaluate_edge_triage(args.gt, args.preds)
    else:
        out = evaluate_acoustic_cuas(args.gt, args.preds)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
