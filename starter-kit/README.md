# Axelera Embedded 110m Starter Kit — challenge-2-rgb-thermal-isr

RF-DETR / CNN starter kit for **Axelera Embedded 110m** (M.2 2280 Key-M PCIe Gen3 x4, 1x Metis AIPU, 214 INT8 TOPS, 1 GB LPDDR4x DRAM, 3.5W-9W).

## Environment Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

Use Python 3.11 or newer. MOT prediction also requires `ffmpeg` on PATH. Training downloads pretrained weights unless disabled where supported.

## Deployment status

The supplied reference runs on CPU ONNX Runtime. Metis compilation, INT8 accuracy, operator placement, memory, and power are not verified. RF-DETR backbone export does not include a runnable host decoder partition.

Follow the [Voyager SDK compiler guide](https://docs.axelera.ai/sdk/v1.8/reference/compiler/compiler-cli/) and check `axcompile --help` for your installed version. Its model input flag is `-i`. Use the train-only calibration folder and supply `--transform preprocess.py`, implementing the SDK's `get_preprocess_transform(image)` callback for your model. It must resize RGB, produce float32 CHW data, scale pixels to [0, 1], and apply ImageNet normalization only for RF-DETR; CNN graphs already include it. The callback is deployment integration work, not a supplied file.

`baseline_metrics.json` contains historical CPU smoke results, not scores guaranteed by a fresh training run or hardware benchmarks.

## Visual Data Explorer

From the challenge root directory (`..`), run the zero-dependency viewer:

```bash
python3 viewer.py
```

## Preprocessing & Export Rules

- **RF-DETR**: `export_rfdetr_axelera.py` traces `RFDETRNano` (`384x384`) expecting ImageNet-normalized NCHW RGB inputs (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- **Name-Indexed ONNX Outputs**: Always resolve outputs by tensor name (`"dets"` and `"labels"`) since both share shape `[1, 300, 4]` for `num_classes=3`.

## Quickstart

```bash
python train_rfdetr.py --dataset-dir ../dataset --model nano --epochs 25 --output-dir runs/c2_nano
python export_rfdetr_axelera.py --model nano --dataset-dir ../dataset \
  --weights runs/c2_nano/checkpoint_best_total.pth \
  --calibration-dir ../calibration
python predict_onnx_baseline.py detect --onnx exports/rfdetr/full/inference_model.onnx \
  --dataset-dir ../dataset --split valid --out c2_preds.json
python evaluate_submissions.py detect --gt ../dataset/valid/_annotations.coco.json --preds c2_preds.json
python predict_onnx_baseline.py mot --onnx exports/rfdetr/full/inference_model.onnx \
  --eval-dir ../streams_eval --out-dir pred_mot/
python evaluate_submissions.py mot --gt-dir ../streams_eval --pred-dir pred_mot/ --split valid
```

## Submission Formats

- **COCO Detection (`preds.json`)**: JSON list of `{"image_id": int, "category_id": 0-based int, "bbox": [x, y, w, h], "score": float in [0, 1]}` (scored across both `rgb` and `thermal` subsets).
- **MOTChallenge (`pred_mot/seq_XXXX.txt`)**: Comma-separated `frame_1based,track_id,x,y,w,h,conf,class_id_1based,1.0` across `../streams_eval`.
