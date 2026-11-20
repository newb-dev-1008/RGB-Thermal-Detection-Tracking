# Challenge 2: Day RGB + Thermal IR ISR Detection & Multi-Stream Tracking

**Presentation:** [Two-slide PDF](presentation/challenge-slides.pdf) · [Editable PowerPoint](presentation/challenge-slides.pptx). Includes a challenge overview and real validation examples from this data pack.

> **Target Platform**: **Axelera Embedded 110m** — M.2 2280 Key-M (PCIe Gen3 x4), 1x Metis AIPU (214 TOPS INT8, 3.5–9 W, **1 GB LPDDR4x** on-card DRAM) connected to a host processor.
> **Recommended Detector**: **Roboflow RF-DETR** (`RFDETRNano` at `384x384`, ~30.2M params, Apache-2.0). YOLO / `ultralytics` are **not** used in this challenge.

---

## 1. Mission & Problem Statement

Modern ISR pods switch continuously between **Daylight Electro-Optical RGB** and **Long-Wave Infrared (Thermal IR)** sensors. A single edge accelerator with **1 GB DRAM** cannot afford two independent bloated detector pipelines, yet naive RGB-pretrained detectors struggle severely on thermal imagery (observe our 1-epoch smoke test below: `rgb_mAP50 = 0.5263` vs. `thermal_mAP50 = 0.1799`).

Design a **shared or unified-representation `RF-DETR` detector + tracker** compiled for Metis INT8 that maintains high accuracy across **both** Day RGB and Thermal IR without modality collapse or INT8 quantization degradation.

---

## 2. Dataset Specification

- **Classes (0-indexed COCO IDs)**:
  - `0`: `armored_vehicle`
  - `1`: `vehicle` (unarmored trucks, pickups, UV/transport vehicles)
  - `2`: `person`
- **Modality Tag**: Every `train` / `valid` image entry in `_annotations.coco.json` carries `"modality": "rgb" | "thermal"` for per-modality evaluation (held-out `test` annotations & modality metadata are stripped in public archives).
- **Stratified Leak-Free Splits**: Both `rgb` and `thermal` AND all three classes (`armored_vehicle`, `vehicle`, `person`) plus clean negatives appear across `train`, `valid`, and `test` with **zero** shared `dedup_root_key` clips, SHA-256 hashes, or `dHash64` near-duplicate frames across splits.

| Split | Total Images | Roots | RGB Images (`armored / vehicle / person` boxes) | Thermal Images (`armored / vehicle / person` boxes) |
|---|---:|---:|---|---|
| `train/` | 712 | 13 | 308 (`221 / 63 / 167`) | 404 (`177 / 203 / 75`) |
| `valid/` | 484 | 9 | 253 (`116 / 111 / 300`) | 231 (`143 / 64 / 64`) |
| `test/` | 424 | 9 | — (held-out) | — (held-out) |

- **INT8 Calibration Set (`calibration/`)**: `200` `384x384` images sampled **strictly from `train/`**, stratified **100 RGB + 100 Thermal**.
- **Multi-Stream Evaluation (`streams_eval/`)**: `8` `30 fps` H.264 MP4 streams (`seq_0001`–`seq_0004` **Thermal**; `seq_0005`–`seq_0008` **RGB**; `72`–`150` frames each): `4` `valid` streams (`737` total GT boxes across `seq_0001`, `seq_0002`, `seq_0005`, `seq_0006`) and `4` `test` streams (`gt/gt.txt` held out in public archives).

### Interactive Data Explorer (`viewer.py`)

Browse RGB vs. Thermal IR frames side-by-side, toggle **Thermal Contrast Boost / Polarity Invert**, filter streams by modality, and overlay COCO/MOT predictions offline:

```bash
python3 viewer.py
# Optional prediction overlay + custom port for SSH forwarding (-L 8766:127.0.0.1:8766):
python3 viewer.py --preds starter-kit/c2_preds.json --pred-dir starter-kit/pred_mot --port 8766
```

---

## 3. Baseline & Optimization Angles

1. **Train `RFDETRNano` (`384x384`)**:
   ```bash
   cd starter-kit
   uv run --with "rfdetr[onnx,train]" --with onnx --with onnxscript --with onnxruntime --with scipy \
     python train_rfdetr.py --dataset-dir ../dataset --model nano --epochs 25 --output-dir runs/c2_nano
   ```
2. **Export static ONNX (`opset=17`, `384x384`) & evaluate per-modality breakdown**:
   ```bash
   uv run --with "rfdetr[onnx,train]" --with onnx --with onnxscript --with onnxruntime \
     python export_rfdetr_axelera.py --model nano --num-classes 3 \
     --weights runs/c2_nano/checkpoint_best_total.pth \
     --calibration-dir ../calibration
   ```
   The full ONNX baseline runs on CPU ORT. A Metis backbone plus host decoder requires additional integration; no automatic partition or compiled hardware baseline is supplied. Follow `starter-kit/README.md` for SDK guidance and use ImageNet-normalized calibration inputs. `evaluate_submissions.py detect` automatically detects `"modality"` on `valid/` and reports `rgb_mAP50` and `thermal_mAP50`.

### Smoke-Test Reference (1-Epoch `RFDETRNano` on `valid/` Static ONNX Runtime CPU)

| Metric | 1-Epoch Smoke Result | Notes |
|---|---:|---|
| Val `mAP@50` (PyTorch Trainer) | **0.3475** | `mAP@50:95 = 0.2478`, `F1 = 0.3394` |
| Static ONNX Overall (`valid/`) | `mAP50 = 0.3489`, `mAP50:95 = 0.2487` | `39.54 ms/frame` CPU ORT (`484` imgs) |
| Static ONNX **RGB vs. Thermal** | `rgb_mAP50 = 0.5263`, `thermal_mAP50 = 0.1799` | Strong IR adaptation headroom (`thermal_mAP50:95 = 0.1255`) |
| Static ONNX + IoU Tracker (`valid` `streams_eval/`) | `MOTA = -0.0760`, `IDF1 = 0.2353` | Continuity `0.2055`, `9` ID switches over `737` valid GT boxes |

### Ideas to Beat the Baseline
- **IR Photometric Adaptation**: Random contrast inversion / CLAHE / thermal palette jitter during training so COCO RGB backbones bridge to LWIR rapidly.
- **Class-Balanced Sampling & Mosaic/Mix**: Up-weight rare RGB `vehicle` (`63` train boxes) and Thermal `person` (`75` train boxes) tails cleanly.
- **Mixed RGB+IR INT8 Calibration**: Verify quantization sensitivity on both dark thermal silhouettes and high-contrast sunlit armor using the 100/100 `calibration/` set.

---

## 4. Metrics and Proposed Judging Weights

The scorer returns individual metrics, not a combined leaderboard score. The percentages below are a proposed judging rubric. Organizers must publish the aggregation, normalization, tie-break rules, and hardware measurement protocol before using this dataset for a ranked event. Local scoring uses `valid/`; final scoring requires organizer-held `test/` labels.

Evaluated via `starter-kit/evaluate_submissions.py` (default `--split valid` on `mot`):
- **Per-Modality Balance (`45%`)**: `rgb_mAP50` and `thermal_mAP50` reported separately; organizers must select a balance formula
- **Multi-Stream Tracking (`35%`)**: `MOTA`, `IDF1`, `track_continuity` on `streams_eval/` (`4` Thermal + `4` RGB streams)
- **Edge Deployment (`20%`)**: Compiled single-model deployment + INT8 accuracy retention + latency & DRAM footprint

This is a small curated pilot with 13 training roots and 9 validation roots. Repeated frames are correlated. RGB and thermal samples are separate scenes, not spatially registered sensor pairs. The MOT scorer is class-agnostic and uses per-frame IoU matching for MOTA and identity-pair counts for IDF1; do not assume exact TrackEval parity.
