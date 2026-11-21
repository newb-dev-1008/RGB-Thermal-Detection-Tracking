"""
Thermal-aware photometric augmentations for bridging RGB ↔ thermal domain gap.

These transforms are applied at the PIL Image level and are composable with
standard torchvision/rfdetr augmentation pipelines.

Key ideas:
- RandomThermalPolarity: flips pixel intensity (white-hot ↔ black-hot palette)
- ThermalCLAHE: contrast-limited adaptive histogram equalization for LWIR images
- RandomThermalPalette: applies a false-color thermal palette (ironbow, jet, rainbow)
- RandomGrayscale3ch: converts RGB to 3-channel grayscale to bridge texture gap
- RandomGaussianNoise: adds sensor-like noise common in uncooled thermal arrays
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Thermal palette LUTs (256-entry RGB lookup tables)
# ---------------------------------------------------------------------------

def _build_ironbow_lut() -> np.ndarray:
    """Ironbow (FLIR-style) false-color LUT: black→blue→red→yellow→white."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.25:
            r, g, b = 0, 0, int(t * 4 * 128)
        elif t < 0.50:
            r, g, b = int((t - 0.25) * 4 * 255), 0, int(128 + (t - 0.25) * 4 * 127)
        elif t < 0.75:
            r, g, b = 255, int((t - 0.50) * 4 * 255), int(255 - (t - 0.50) * 4 * 255)
        else:
            r, g, b = 255, 255, int((t - 0.75) * 4 * 255)
        lut[i] = [r, g, b]
    return lut


def _build_rainbow_lut() -> np.ndarray:
    """Standard rainbow LUT: blue→cyan→green→yellow→red."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.25:
            r, g, b = 0, int(t * 4 * 255), 255
        elif t < 0.50:
            r, g, b = 0, 255, int(255 - (t - 0.25) * 4 * 255)
        elif t < 0.75:
            r, g, b = int((t - 0.50) * 4 * 255), 255, 0
        else:
            r, g, b = 255, int(255 - (t - 0.75) * 4 * 255), 0
        lut[i] = [r, g, b]
    return lut


IRONBOW_LUT = _build_ironbow_lut()
RAINBOW_LUT = _build_rainbow_lut()


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

class RandomThermalPolarity:
    """
    Randomly invert pixel intensities (white-hot ↔ black-hot).
    
    Thermal cameras can display images in either polarity. Training with both
    makes the model robust to the LWIR display mode.
    
    Args:
        p: Probability of applying the inversion.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
            arr = 255 - arr
            return Image.fromarray(arr)
        return img


class ThermalCLAHE:
    """
    Contrast-Limited Adaptive Histogram Equalization for thermal images.
    
    LWIR images often have low global contrast with locally important
    hot/cold regions. CLAHE enhances these local structures.
    
    Args:
        clip_limit: Contrast clip limit (0-255 scale for 8-bit tiles).
        tile_grid_size: Number of tiles in (H, W) for CLAHE.
        p: Probability of applying.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        p: float = 0.5,
    ) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.p = p

    def _clahe_numpy(self, gray: np.ndarray) -> np.ndarray:
        """Pure-numpy CLAHE approximation (tile-based histogram equalization)."""
        h, w = gray.shape
        th, tw = self.tile_grid_size
        tile_h = max(1, h // th)
        tile_w = max(1, w // tw)
        
        result = np.zeros_like(gray, dtype=np.float32)
        count = np.zeros_like(gray, dtype=np.float32)
        
        for ti in range(th):
            for tj in range(tw):
                y0 = ti * tile_h
                y1 = min(h, y0 + tile_h)
                x0 = tj * tile_w
                x1 = min(w, x0 + tile_w)
                tile = gray[y0:y1, x0:x1].astype(np.float32)
                
                # Compute histogram
                hist, bins = np.histogram(tile.flatten(), bins=256, range=(0, 255))
                # Clip histogram
                clip = int(self.clip_limit * tile.size / 256)
                excess = np.sum(np.maximum(0, hist - clip))
                hist = np.minimum(hist, clip)
                hist += excess // 256  # redistribute excess uniformly
                # Cumulative distribution
                cdf = np.cumsum(hist).astype(np.float32)
                cdf_min = cdf[cdf > 0][0] if np.any(cdf > 0) else 1.0
                total = tile.size
                lut = np.clip(
                    (cdf - cdf_min) / max(1, total - cdf_min) * 255.0,
                    0, 255
                ).astype(np.float32)
                
                # Apply LUT
                tile_eq = lut[tile.astype(np.int32).clip(0, 255)]
                result[y0:y1, x0:x1] += tile_eq
                count[y0:y1, x0:x1] += 1.0
        
        return np.clip(result / np.maximum(count, 1.0), 0, 255).astype(np.uint8)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        
        try:
            import cv2  # noqa: F401 — optional speedup
            arr = np.asarray(img.convert("L"), dtype=np.uint8)
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
            )
            eq = clahe.apply(arr)
        except ImportError:
            arr = np.asarray(img.convert("L"), dtype=np.uint8)
            eq = self._clahe_numpy(arr)
        
        # Return as 3-channel image
        eq_rgb = np.stack([eq, eq, eq], axis=-1)
        return Image.fromarray(eq_rgb)


class RandomThermalPalette:
    """
    Apply a false-color thermal palette to a grayscale image.
    
    Thermal cameras often display with false-color palettes (ironbow, rainbow, jet).
    Randomly applying these during training helps the backbone generalize.
    
    Args:
        palettes: List of palette names from {'ironbow', 'rainbow', 'gray'}.
        p: Probability of applying a non-gray palette.
    """

    def __init__(
        self,
        palettes: list[str] | None = None,
        p: float = 0.4,
    ) -> None:
        self.palettes = palettes or ["ironbow", "rainbow", "gray"]
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        
        palette = random.choice(self.palettes)
        gray = np.asarray(img.convert("L"), dtype=np.uint8)
        
        if palette == "ironbow":
            rgb = IRONBOW_LUT[gray]
        elif palette == "rainbow":
            rgb = RAINBOW_LUT[gray]
        else:  # gray
            rgb = np.stack([gray, gray, gray], axis=-1)
        
        return Image.fromarray(rgb.astype(np.uint8))


class RandomGrayscale3ch:
    """
    Convert an RGB image to 3-channel grayscale with probability p.
    
    This bridges the texture gap: RGB has rich color textures while thermal
    is essentially grayscale intensity. Training RGB images as "grayscale"
    sometimes teaches the backbone texture-free shape features.
    
    Args:
        p: Probability of converting to grayscale.
    """

    def __init__(self, p: float = 0.15) -> None:
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            gray = img.convert("L")
            return Image.merge("RGB", [gray, gray, gray])
        return img


class RandomGaussianNoise:
    """
    Add Gaussian noise to simulate uncooled microbolometer noise floor.
    
    Args:
        sigma_range: (min, max) noise standard deviation in [0,255].
        p: Probability of applying noise.
    """

    def __init__(
        self,
        sigma_range: tuple[float, float] = (3.0, 15.0),
        p: float = 0.3,
    ) -> None:
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        sigma = random.uniform(*self.sigma_range)
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


class RandomContrastStretch:
    """
    Apply random min-max contrast stretching to simulate varying thermal ranges.
    
    Args:
        pct_range: Percentile clip range (lo_pct, hi_pct) to stretch.
        p: Probability of applying.
    """

    def __init__(
        self,
        pct_range: tuple[float, float] = (1.0, 99.0),
        p: float = 0.4,
    ) -> None:
        self.pct_range = pct_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        lo = np.percentile(arr, self.pct_range[0])
        hi = np.percentile(arr, self.pct_range[1])
        if hi - lo < 1.0:
            return img
        arr = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Compose helpers
# ---------------------------------------------------------------------------

class ModalityAugment:
    """
    Applies modality-specific augmentation based on an image's modality tag.
    
    Usage::
    
        aug = ModalityAugment(thermal_p=0.5, rgb_p=0.15)
        # In dataset __getitem__:
        img = aug(img, modality='thermal')
    """

    def __init__(
        self,
        thermal_p: float = 0.5,
        rgb_grayscale_p: float = 0.15,
        noise_p: float = 0.3,
        contrast_p: float = 0.4,
    ) -> None:
        self.thermal_transforms = [
            RandomThermalPolarity(p=thermal_p),
            ThermalCLAHE(p=thermal_p),
            RandomThermalPalette(p=0.4),
            RandomContrastStretch(p=contrast_p),
            RandomGaussianNoise(p=noise_p),
        ]
        self.rgb_transforms = [
            RandomGrayscale3ch(p=rgb_grayscale_p),
            RandomGaussianNoise(sigma_range=(1.0, 8.0), p=0.2),
        ]

    def __call__(self, img: Image.Image, modality: str = "rgb") -> Image.Image:
        if modality == "thermal":
            for t in self.thermal_transforms:
                img = t(img)
        else:
            for t in self.rgb_transforms:
                img = t(img)
        return img


def build_thermal_transform_list(
    thermal_p: float = 0.5,
    for_thermal: bool = True,
) -> list:
    """Return a list of augmentation callables for composing."""
    if for_thermal:
        return [
            RandomThermalPolarity(p=thermal_p),
            ThermalCLAHE(p=thermal_p),
            RandomThermalPalette(p=0.4),
            RandomContrastStretch(p=0.4),
            RandomGaussianNoise(sigma_range=(3.0, 15.0), p=0.3),
        ]
    else:
        return [
            RandomGrayscale3ch(p=0.15),
            RandomGaussianNoise(sigma_range=(1.0, 8.0), p=0.2),
        ]
