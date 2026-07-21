"""Evaluate pseudo-masks against TotalSegmentator ground-truth using IoU.

Usage:
    python scripts/evaluate_masks.py
    python scripts/evaluate_masks.py --masks_dir data/totalseg_masks
    python scripts/evaluate_masks.py --threshold 0.5
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.labels.templates import CLASS_NAMES

DEFAULT_MASKS_DIR = Path("data/totalseg_masks")


def _compute_iou(pred: np.ndarray, gt: np.ndarray) -> float | None:
    """Compute IoU between two binary arrays. Returns None if GT is empty."""
    if not gt.any():
        return None
    intersection = (pred & gt).sum()
    union        = (pred | gt).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks_dir", default=str(DEFAULT_MASKS_DIR))
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold to binarise soft pseudo-masks (default 0.5)")
    args = parser.parse_args()

    masks_dir = Path(args.masks_dir)
    meta_path = masks_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found — run generate_masks.py first")

    with open(meta_path) as f:
        metadata = json.load(f)

    per_class_ious: dict[int, list[float]] = defaultdict(list)

    for entry in metadata:
        pseudo_mask = np.load(entry["mask_path"])
        gt_mask     = np.load(entry["gt_mask_path"])

        pred_binary = pseudo_mask >= args.threshold
        gt_binary   = gt_mask > 0.5

        iou = _compute_iou(pred_binary, gt_binary)
        if iou is not None:
            per_class_ious[entry["true_label"]].append(iou)

    print(f"\n{'Class':<15} {'N':>6}  {'IoU':>6}")
    print("-" * 30)
    all_ious = []
    results = {}
    for class_idx in range(len(CLASS_NAMES)):
        ious = per_class_ious.get(class_idx, [])
        mean_iou = float(np.mean(ious)) if ious else float("nan")
        results[CLASS_NAMES[class_idx]] = {"n": len(ious), "iou": mean_iou}
        all_ious.extend(ious)
        print(f"{CLASS_NAMES[class_idx]:<15} {len(ious):>6}  {mean_iou:>6.4f}")

    miou = float(np.mean(all_ious)) if all_ious else float("nan")
    results["mIoU"] = miou
    print("-" * 30)
    print(f"{'mIoU':<15} {'':>6}  {miou:>6.4f}")

    out_path = masks_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
