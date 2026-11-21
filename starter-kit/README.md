# RGB + Thermal ISR Detection & Tracking — Starter Kit

RF-DETR dual-modality starter kit for the **Axelera Embedded 110m**  
(M.2 2280 Key-M PCIe Gen3 x4 · 1× Metis AIPU · 214 INT8 TOPS · 1 GB LPDDR4x · 3.5–9 W).

Three classes: `armored_vehicle` (0) · `vehicle` (1) · `person` (2)  
Two modalities in the same dataset: **Day RGB** and **Long-Wave Infrared (Thermal)**

---

## Repository layout

```
starter-kit/
├── train_rfdetr.py              # Baseline single-pass trainer (original)
├── train_dual_modality.py       # Modality-aware trainer — USE THIS
├── augmentations_thermal.py     # IR photometric augmentations (polarity, CLAHE, palette …)
├── balanced_dataset.py          # Inverse-frequency class/modality sampler
├── export_rfdetr_axelera.py     # ONNX export + Axelera axcompile guide
├── predict_onnx_baseline.py     # Baseline greedy-IoU predictor
├── predict_onnx_improved.py     # Kalman-filter tracker + per-modality thresholds
├── evaluate_submissions.py      # Official scorer (COCO mAP + MOT metrics)
└── preprocess.py                # Axelera Voyager SDK preprocess callback
```

---

## 1 · Environment setup

> Requires **Python ≥ 3.11** and [`uv`](https://github.com/astral-sh/uv).  
> `ffmpeg` must be on `PATH` for MOT video decoding.

```bash
# From the starter-kit/ directory:
uv venv
source .venv/bin/activate        # Linux / macOS / WSL
# .venv\Scripts\activate         # Windows PowerShell

uv pip install -e .
```

The `pyproject.toml` pulls in `rfdetr`, `onnxruntime`, `torch`, `scipy`, `pillow`, and `numpy`.

---

## 2 · Explore the data

```bash
# From the challenge root (one level up):
python viewer.py
# With prediction overlay:
python viewer.py --preds starter-kit/c2_preds.json --pred-dir starter-kit/pred_mot
```

Inspect the class/modality balance before training:

```bash
python balanced_dataset.py ../dataset/train/_annotations.coco.json
```

---

## 3 · Training

### 3a — Dual-modality training (recommended)

`train_dual_modality.py` adds three improvements on top of the baseline:

| Feature | Detail |
|---|---|
| **Thermal augmentations** | Polarity flip · CLAHE · false-colour palette · contrast stretch · Gaussian noise |
| **Class-balanced sampling** | Inverse-frequency weights — rare combos (RGB `vehicle`, thermal `person`) upsampled ~3–7× |
| **Harmonic early-stopping** | Monitors `2·rgb_mAP50·thermal_mAP50 / (rgb+thermal)` instead of overall mAP to prevent modality collapse |

```bash
uv run --with "rfdetr[onnx,train]" --with onnx --with onnxscript \
       --with onnxruntime --with scipy \
  python train_dual_modality.py \
    --dataset-dir ../dataset \
    --model nano \
    --epochs 50 \
    --batch-size 8 \
    --grad-accum 2 \
    --lr 1e-4 \
    --thermal-aug-p 0.5 \
    --rgb-grayscale-p 0.15 \
    --early-stop-metric harmonic \
    --early-stop-patience 10 \
    --output-dir runs/c2_dual
```

**Key flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--thermal-aug-p` | `0.5` | Probability for each thermal augmentation |
| `--rgb-grayscale-p` | `0.15` | Probability of converting an RGB image to 3-ch grayscale |
| `--early-stop-metric` | `harmonic` | `harmonic` (recommended) · `overall` · `thermal` |
| `--no-balanced-sampling` | off | Disables weighted sampler (use vanilla ordering) |

After training, the script automatically exports a temporary ONNX and prints per-modality mAP50 results. Checkpoints are saved to `runs/c2_dual/`.

### 3b — Baseline trainer (original, for comparison)

```bash
python train_rfdetr.py \
  --dataset-dir ../dataset \
  --model nano \
  --epochs 25 \
  --output-dir runs/c2_nano
```

---

## 4 · Export to ONNX

Produces a **static-shape opset-17 ONNX** model (`384×384`, batch=1) and prints the full `axcompile` command for the Axelera Metis AIPU.

```bash
python export_rfdetr_axelera.py \
  --model nano \
  --num-classes 3 \
  --weights runs/c2_dual/checkpoint_best_total.pth \
  --calibration-dir ../calibration \
  --mode both \
  --output-dir exports/rfdetr
```

Outputs:

| File | Used for |
|---|---|
| `exports/rfdetr/full/inference_model.onnx` | CPU ONNX Runtime evaluation (steps 5 & 6) |
| `exports/rfdetr/backbone/inference_model.onnx` | Axelera Metis AIPU compilation (step 7) |

> **Output naming**: always index ONNX outputs by **name** (`"dets"`, `"labels"`), not by position — both tensors share shape `[1, 300, 4]`.

---

## 5 · Evaluate detection (COCO mAP)

```bash
# Improved predictor — identical output format to baseline
python predict_onnx_improved.py detect \
  --onnx exports/rfdetr/full/inference_model.onnx \
  --dataset-dir ../dataset \
  --split valid \
  --out c2_preds_dual.json

python evaluate_submissions.py detect \
  --gt ../dataset/valid/_annotations.coco.json \
  --preds c2_preds_dual.json
```

Expected output fields: `mAP50`, `mAP50_95`, `rgb_mAP50`, `thermal_mAP50`, `small_AP50`.

---

## 6 · Evaluate multi-object tracking (MOT)

The improved predictor uses a **Kalman-filter + Hungarian** tracker with per-modality score thresholds. Thermal sequences automatically receive a lower threshold because LWIR images produce lower model confidence.

```bash
python predict_onnx_improved.py mot \
  --onnx exports/rfdetr/full/inference_model.onnx \
  --eval-dir ../streams_eval \
  --out-dir pred_mot_dual/ \
  --score-thr-rgb 0.25 \
  --score-thr-thermal 0.15 \
  --iou-thr 0.30 \
  --max-age 5

python evaluate_submissions.py mot \
  --gt-dir ../streams_eval \
  --pred-dir pred_mot_dual/ \
  --split valid
```

Output metrics: `MOTA`, `IDF1`, `track_continuity`, `id_switches`.

**Tuning tips:**

| Parameter | Lower → | Higher → |
|---|---|---|
| `--score-thr-thermal` | More detections (better recall) | Fewer false positives |
| `--iou-thr` | Looser matching (fewer ID-switches) | Stricter, more switches |
| `--max-age` | Tracks pruned faster | Longer re-ID window (less fragmentation) |

---

## 7 · Edge deployment — Axelera Embedded 110m (Raspberry Pi host)

The Axelera Embedded 110m is an **M.2 2280 Key-M card** inserted into a Raspberry Pi 5 or CM5 carrier via PCIe. The Metis AIPU runs the compiled INT8 backbone; the host CPU runs the transformer decoder and tracker.

### 7a — Host setup (Raspberry Pi)

```bash
# Install Axelera Voyager SDK on the Pi (follow official guide):
# https://docs.axelera.ai/sdk/v1.8/getting-started/

# Confirm the card is visible:
axelera-info          # should list 1x Metis device

# Install Python runtime dependencies (no torch needed on the Pi):
pip install onnxruntime-openvino pillow numpy scipy
# or the Axelera runtime wheel if provided by the SDK
```

### 7b — INT8 compilation (run on a development machine, not the Pi)

The `axcompile` tool runs on x86 Linux. Cross-compile the backbone, then copy the output to the Pi.

```bash
# 1. Activate the Voyager SDK environment (follow SDK install guide)
source /opt/axelera/voyager-sdk/setup.sh

# 2. Compile the backbone (not the full model — decoder runs on host CPU)
axcompile \
  -i exports/rfdetr/backbone/inference_model.onnx \
  --imageset ../calibration \
  --transform preprocess.py \
  --input-shape 1,3,384,384 \
  -o exports/rfdetr/metis_axm/

# The calibration/ folder contains 100 RGB + 100 Thermal images.
# preprocess.py implements the get_preprocess_transform() callback:
#   PIL image → resize 384×384 → float32 CHW [0,1] → ImageNet norm
# This same pipeline is used for BOTH RGB and thermal inputs.

# 3. Copy compiled artefact to the Pi:
scp -r exports/rfdetr/metis_axm/ pi@raspberrypi.local:~/c2_model/
```

> **Memory budget**: the full ONNX backbone is ~89 MB; INT8 compression targets < 30 MB on-card DRAM. The transformer decoder stays on the Pi's ARM CPU and uses standard ONNX Runtime.

### 7c — Runtime inference on the Pi

The Metis AIPU runs the backbone; ONNX Runtime runs the decoder on the Pi CPU. The `predict_onnx_improved.py` predictor targets CPU ONNX Runtime and can be adapted for the split-inference pattern:

```python
# Pseudocode for Metis + host-decoder split inference
import axelera_runtime as ax       # Voyager SDK Python bindings
import onnxruntime as ort
from preprocess import get_preprocess_transform
from PIL import Image

# Load backbone on Metis AIPU
backbone = ax.load("~/c2_model/metis_axm/")
# Load decoder on host CPU (exported with backbone_only=False → "full" ONNX,
# but only the decoder sub-graph is needed here — see SDK partition guide)
decoder_sess = ort.InferenceSession("exports/rfdetr/full/inference_model.onnx",
                                     providers=["CPUExecutionProvider"])

for frame in video_stream:
    # Preprocess (same for RGB and thermal)
    chw = get_preprocess_transform(frame)          # (3, 384, 384) float32
    # Backbone on AIPU (INT8, ~5 ms on Metis)
    features = backbone.run(chw[None])             # (1, 256, 24, 24)
    # Decoder on host CPU (~35 ms on Pi 5)
    dets, labels = decoder_sess.run(["dets", "labels"], {"features": features})
    # Post-process with Kalman tracker (see predict_onnx_improved.py)
    ...
```

> **Note**: the host-decoder partition and Axelera Python bindings require the full Voyager SDK. The exact API depends on your installed SDK version — refer to the [Voyager Runtime Guide](https://docs.axelera.ai/sdk/v1.8/reference/runtime/).

### 7d — Preprocessing rules (critical for INT8 accuracy)

Both RGB and thermal images go through the **same** pipeline:

```
PIL.open(path).convert("RGB")
  → resize to 384×384 (BILINEAR)
  → float32 / 255.0
  → subtract [0.485, 0.456, 0.406]
  → divide by  [0.229, 0.224, 0.225]
  → shape: (1, 3, 384, 384) NCHW
```

Equivalent in pixel scale (for SDK `pixel_mean`/`pixel_std` flags):

```
pixel_mean = [123.675, 116.28,  103.53 ]
pixel_std  = [ 58.395,  57.12,   57.375]
```

The `preprocess.py` callback already implements this. Verify it on a sample:

```bash
python preprocess.py ../dataset/train/c2_0001.jpg
```

---

## 8 · Submission formats

| Output | Format |
|---|---|
| **COCO detection** (`c2_preds_dual.json`) | JSON list · `{"image_id": int, "category_id": 0-based int, "bbox": [x,y,w,h], "score": float}` |
| **MOT tracking** (`pred_mot_dual/seq_XXXX.txt`) | CSV · `frame,track_id,x,y,w,h,conf,class_id_1based,1.0` (1-based frame index) |

---

## 9 · Baseline vs improved — reference metrics

| Metric | Baseline (1 epoch) | Target (50 epochs + dual) |
|---|---|---|
| `rgb_mAP50` | 0.5263 | ≥ 0.60 |
| `thermal_mAP50` | 0.1799 | ≥ 0.45 |
| `MOTA` | −0.076 | ≥ 0.20 |
| `IDF1` | 0.235 | ≥ 0.40 |
| `track_continuity` | 0.206 | ≥ 0.50 |

Full historical CPU smoke results: [`baseline_metrics.json`](baseline_metrics.json).

---

## Quick reference

```bash
# Full pipeline — dual modality, Kalman tracker
python train_dual_modality.py --dataset-dir ../dataset --model nano --epochs 50 --output-dir runs/c2_dual
python export_rfdetr_axelera.py --model nano --num-classes 3 --weights runs/c2_dual/checkpoint_best_total.pth --calibration-dir ../calibration
python predict_onnx_improved.py detect --onnx exports/rfdetr/full/inference_model.onnx --dataset-dir ../dataset --split valid --out c2_preds_dual.json
python evaluate_submissions.py detect --gt ../dataset/valid/_annotations.coco.json --preds c2_preds_dual.json
python predict_onnx_improved.py mot --onnx exports/rfdetr/full/inference_model.onnx --eval-dir ../streams_eval --out-dir pred_mot_dual/ --score-thr-thermal 0.15 --max-age 5
python evaluate_submissions.py mot --gt-dir ../streams_eval --pred-dir pred_mot_dual/ --split valid
```
