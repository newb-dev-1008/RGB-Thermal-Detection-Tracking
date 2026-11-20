"""Fine-tune Roboflow RF-DETR (Nano/Small/Medium) on Challenge 1 or Challenge 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rfdetr import RFDETRMedium, RFDETRNano, RFDETRSmall

MODELS = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to challenge dataset/ containing train/, valid/, and test/ with _annotations.coco.json",
    )
    parser.add_argument("--model", choices=tuple(MODELS), default="nano")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/rfdetr"))
    parser.add_argument("--run-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_json = args.dataset_dir / "train" / "_annotations.coco.json"
    if not train_json.exists():
        raise FileNotFoundError(f"Missing RF-DETR annotation file: {train_json}")

    cats = json.loads(train_json.read_text(encoding="utf-8")).get("categories", [])
    if not cats:
        raise ValueError(f"Empty categories list in {train_json}")
    class_names = [c["name"] for c in sorted(cats, key=lambda c: int(c["id"]))]
    print(f"Training RFDETR-{args.model} on {args.dataset_dir} classes={class_names}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = MODELS[args.model]()
    model.train(
        dataset_dir=str(args.dataset_dir),
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
        early_stopping=True,
        run_test=args.run_test,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()
