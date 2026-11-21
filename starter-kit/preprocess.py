"""
Axelera Voyager SDK preprocess callback for RF-DETR (RGB or Thermal inputs).

This file implements the ``get_preprocess_transform(image)`` callback expected
by ``axcompile --transform preprocess.py``.

The callback must:
1. Resize the input image to 384×384 (RF-DETR Nano input resolution)
2. Convert to float32 CHW in [0, 1]
3. Apply ImageNet normalization (mean/std)

This exact pipeline applies to BOTH RGB and Thermal modalities because:
- Thermal images are stored as standard JPEG/PNG files
- The model was trained with thermal images converted to RGB (PIL.convert("RGB"))
- ImageNet normalization is applied uniformly — thermal "RGB" pixel values
  are treated as generic 3-channel inputs, not color-calibrated values

The calibration/ folder contains 100 RGB + 100 Thermal images for INT8
quantization. Use all 200 for mixed-modality INT8 calibration:

    axcompile -i exports/rfdetr/backbone/inference_model.onnx \\
        --imageset ../calibration \\
        --transform preprocess.py \\
        --input-shape 1,3,384,384 \\
        -o exports/rfdetr/metis_axm/

For pure-CPU ONNX Runtime inference, this normalization is already built into
export_rfdetr_axelera.py and predict_onnx_improved.py.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# RF-DETR Nano input resolution
INPUT_SIZE = 384

# ImageNet normalization constants (same for both RGB and thermal inputs)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_preprocess_transform(image: "Image.Image | np.ndarray | str") -> np.ndarray:
    """
    Axelera SDK preprocess callback.

    Accepts a PIL Image, a NumPy HWC uint8 array, or a file path string.
    Returns a float32 CHW array with shape (3, 384, 384), normalized by
    ImageNet mean/std, suitable for RF-DETR ONNX backbone input.

    Args:
        image: Input image (PIL Image, numpy HWC uint8, or path string).

    Returns:
        np.ndarray of shape (3, 384, 384), dtype float32.
    """
    # Normalize to PIL Image
    if isinstance(image, (str, bytes)):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        image = Image.fromarray(image)

    # Resize and convert to RGB (this handles thermal grayscale JPEGs too)
    rgb = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)

    # HWC → float [0, 1] → CHW
    arr = np.asarray(rgb, dtype=np.float32) / 255.0  # (H, W, 3)
    chw = np.transpose(arr, (2, 0, 1))               # (3, H, W)

    # ImageNet normalization: (x - mean) / std per channel
    chw = (chw - _MEAN[:, None, None]) / _STD[:, None, None]
    return chw.astype(np.float32)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    test_path = None
    if len(sys.argv) > 1:
        test_path = sys.argv[1]

    if test_path:
        out = get_preprocess_transform(test_path)
        print(f"Input: {test_path}")
        print(f"Output shape: {out.shape}, dtype: {out.dtype}")
        print(f"Value range: [{out.min():.3f}, {out.max():.3f}]")
        print(f"Mean per channel: {out.mean(axis=(1,2))}")
        print("OK: preprocess callback works correctly.")
    else:
        # Create a synthetic test
        dummy_rgb = Image.fromarray(
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        )
        dummy_thermal = Image.fromarray(
            np.random.randint(50, 200, (352, 640), dtype=np.uint8).astype(np.uint8)
        )
        for name, img in [("RGB", dummy_rgb), ("Thermal (grayscale)", dummy_thermal)]:
            out = get_preprocess_transform(img)
            assert out.shape == (3, 384, 384), f"Wrong shape: {out.shape}"
            assert out.dtype == np.float32, f"Wrong dtype: {out.dtype}"
            print(f"{name}: shape={out.shape} range=[{out.min():.2f}, {out.max():.2f}] ✓")
        print("\nAll preprocess tests passed.")
