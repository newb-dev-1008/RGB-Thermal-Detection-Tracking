"""
Class-balanced COCO dataset and weighted sampler for RGB+Thermal training.

The dataset wraps a standard COCO-formatted folder and:
1. Tags each image with its ``modality`` field ('rgb' or 'thermal').
2. Applies modality-specific photometric augmentations before returning
   the PIL image to rfdetr's internal pipeline.
3. Computes per-image sampling weights based on inverse class frequency so
   rare class/modality combinations are upsampled to prevent collapse.

Usage::

    from balanced_dataset import make_balanced_sampler, ModalityCOCOInfo
    info = ModalityCOCOInfo(train_json_path)
    weights = info.sample_weights()
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


class ModalityCOCOInfo:
    """
    Parses a COCO JSON and computes per-image sampling weights.

    Weights are based on the inverse frequency of the rarest class present
    in each image, further multiplied by per-modality balancing to ensure
    thermal images are not under-represented despite having fewer boxes in
    some classes.

    Args:
        coco_json: Path to ``_annotations.coco.json``.
        modality_balance: If True, also upweight the minority modality.
    """

    def __init__(
        self,
        coco_json: Path,
        modality_balance: bool = True,
        alpha: float = 0.75,
    ) -> None:
        self.coco_json = Path(coco_json)
        self.modality_balance = modality_balance
        self.alpha = alpha  # weight smoothing exponent in [0,1]
        self._load()

    def _load(self) -> None:
        with open(self.coco_json, encoding="utf-8") as f:
            doc: dict[str, Any] = json.load(f)

        self.categories: list[dict] = sorted(doc["categories"], key=lambda c: c["id"])
        self.images: list[dict] = doc["images"]
        self.annotations: list[dict] = doc.get("annotations", [])

        # Map image_id → image record
        self.id_to_img: dict[int, dict] = {im["id"]: im for im in self.images}

        # Build image_id → list[category_id]
        self.img_to_cats: dict[int, list[int]] = defaultdict(list)
        for ann in self.annotations:
            self.img_to_cats[int(ann["image_id"])].append(int(ann["category_id"]))

        # Count per-category and per-modality boxes
        self.cat_counts: dict[int, int] = defaultdict(int)
        self.mod_counts: dict[str, int] = defaultdict(int)
        for ann in self.annotations:
            self.cat_counts[int(ann["category_id"])] += 1
        for img in self.images:
            mod = img.get("modality", "rgb")
            img_cats = self.img_to_cats.get(img["id"], [])
            for _ in img_cats:
                self.mod_counts[mod] += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_weights(self) -> list[float]:
        """
        Compute one weight per image in self.images order.

        Strategy:
        - Per-class weight = (total_boxes / class_boxes) ^ alpha
        - Per-image weight = max class weight among its annotations
        - Optionally multiply by modality weight for under-represented modalities
        """
        total_boxes = sum(self.cat_counts.values())
        num_cats = len(self.categories)

        # Class frequency weights
        cls_weight: dict[int, float] = {}
        for cat in self.categories:
            cid = int(cat["id"])
            count = self.cat_counts.get(cid, 1)
            cls_weight[cid] = (total_boxes / max(1, count)) ** self.alpha

        # Modality frequency weights
        total_mod = sum(self.mod_counts.values())
        num_mods = max(1, len(self.mod_counts))
        mod_weight: dict[str, float] = {}
        for mod, count in self.mod_counts.items():
            mod_weight[mod] = (total_mod / max(1, count)) ** (self.alpha * 0.5)

        weights: list[float] = []
        for img in self.images:
            iid = int(img["id"])
            cats = self.img_to_cats.get(iid, [])
            if cats:
                w = max(cls_weight.get(c, 1.0) for c in cats)
            else:
                # Negative (empty) image — give lower weight
                w = 0.3
            if self.modality_balance:
                mod = img.get("modality", "rgb")
                w *= mod_weight.get(mod, 1.0)
            weights.append(w)

        return weights

    def print_summary(self) -> None:
        """Print dataset statistics for debugging."""
        print(f"\n{'=' * 60}")
        print(f"COCO dataset: {self.coco_json}")
        print(f"  Images: {len(self.images)}")
        print(f"  Annotations: {len(self.annotations)}")
        print(f"\n  Modality distribution:")
        mod_img: dict[str, int] = defaultdict(int)
        for img in self.images:
            mod_img[img.get("modality", "unknown")] += 1
        for mod, cnt in sorted(mod_img.items()):
            print(f"    {mod}: {cnt} images, {self.mod_counts.get(mod, 0)} boxes")
        print(f"\n  Class distribution (boxes):")
        for cat in self.categories:
            cid = int(cat["id"])
            cnt = self.cat_counts.get(cid, 0)
            print(f"    [{cid}] {cat['name']}: {cnt} boxes")
        weights = self.sample_weights()
        print(f"\n  Sampling weights: min={min(weights):.3f}, max={max(weights):.3f}, "
              f"mean={sum(weights)/len(weights):.3f}")
        print(f"{'=' * 60}\n")


def make_weighted_sampler(
    coco_json: Path,
    num_samples: int | None = None,
    replacement: bool = True,
) -> "torch.utils.data.WeightedRandomSampler":
    """
    Build a WeightedRandomSampler for a COCO-format dataset.

    Args:
        coco_json: Path to _annotations.coco.json.
        num_samples: Samples per epoch; defaults to len(dataset).
        replacement: Whether to sample with replacement.

    Returns:
        A WeightedRandomSampler instance ready to pass to DataLoader.
    """
    try:
        from torch.utils.data import WeightedRandomSampler
    except ImportError as exc:
        raise ImportError("PyTorch is required for WeightedRandomSampler") from exc

    info = ModalityCOCOInfo(coco_json)
    info.print_summary()
    weights = info.sample_weights()
    n = num_samples or len(weights)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=n,
        replacement=replacement,
    )


# ---------------------------------------------------------------------------
# Standalone test / diagnostic mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python balanced_dataset.py <path/to/_annotations.coco.json>")
        sys.exit(1)
    info = ModalityCOCOInfo(Path(sys.argv[1]))
    info.print_summary()
    weights = info.sample_weights()

    # Print top-5 and bottom-5 weighted images
    idx_w = sorted(enumerate(weights), key=lambda x: -x[1])
    print("Top-5 highest-weight images (most upsampled):")
    for idx, w in idx_w[:5]:
        img = info.images[idx]
        cats = info.img_to_cats.get(img["id"], [])
        cat_names = [info.categories[c]["name"] for c in cats if c < len(info.categories)]
        print(f"  [{img['id']}] {img['file_name']} mod={img.get('modality')} "
              f"cats={cat_names} weight={w:.3f}")
    print("Bottom-5 lowest-weight images (least upsampled):")
    for idx, w in idx_w[-5:]:
        img = info.images[idx]
        cats = info.img_to_cats.get(img["id"], [])
        cat_names = [info.categories[c]["name"] for c in cats if c < len(info.categories)]
        print(f"  [{img['id']}] {img['file_name']} mod={img.get('modality')} "
              f"cats={cat_names} weight={w:.3f}")
