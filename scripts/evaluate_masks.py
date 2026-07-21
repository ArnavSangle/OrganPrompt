"""Evaluate pseudo-masks against TotalSegmentator ground-truth.

Metrics: IoU, HD95, Dice, Sensitivity, Precision, RVE, NSD

Usage:
    python scripts/evaluate_masks.py
    python scripts/evaluate_masks.py --masks_dir data/totalseg_masks
    python scripts/evaluate_masks.py --threshold 0.5 --nsd_tolerance 3
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.labels.templates import CLASS_NAMES

DEFAULT_MASKS_DIR = Path("data/totalseg_masks")


def _compute_iou(pred: np.ndarray, gt: np.ndarray) -> float | None:
    if not gt.any():
        return None
    intersection = (pred & gt).sum()
    union        = (pred | gt).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def _compute_dice(pred: np.ndarray, gt: np.ndarray) -> float | None:
    if not gt.any():
        return None
    intersection = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2 * intersection) / float(denom) if denom > 0 else 0.0


def _compute_hd95(pred: np.ndarray, gt: np.ndarray) -> float | None:
    if not gt.any() or not pred.any():
        return None
    pred_surface = pred ^ binary_erosion(pred)
    gt_surface   = gt   ^ binary_erosion(gt)
    d_pred = distance_transform_edt(~pred)[gt_surface]
    d_gt   = distance_transform_edt(~gt)[pred_surface]
    return float(np.percentile(np.concatenate([d_pred, d_gt]), 95))


def _compute_sensitivity(pred: np.ndarray, gt: np.ndarray) -> float | None:
    if not gt.any():
        return None
    tp = (pred & gt).sum()
    fn = (~pred & gt).sum()
    return float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0


def _compute_precision(pred: np.ndarray, gt: np.ndarray) -> float | None:
    if not gt.any():
        return None
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    return float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0


def _compute_rve(pred: np.ndarray, gt: np.ndarray) -> float | None:
    if not gt.any():
        return None
    gt_vol = float(gt.sum())
    return float(abs(pred.sum() - gt_vol)) / gt_vol if gt_vol > 0 else 0.0


def _compute_nsd(pred: np.ndarray, gt: np.ndarray, tolerance: float) -> float | None:
    if not gt.any() or not pred.any():
        return None
    pred_surface = pred ^ binary_erosion(pred)
    gt_surface   = gt   ^ binary_erosion(gt)
    # fraction of each surface within tolerance of the other
    pred_within = (distance_transform_edt(~gt_surface)[pred_surface] <= tolerance).sum()
    gt_within   = (distance_transform_edt(~pred_surface)[gt_surface] <= tolerance).sum()
    denom = pred_surface.sum() + gt_surface.sum()
    return float(pred_within + gt_within) / float(denom) if denom > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks_dir", default=str(DEFAULT_MASKS_DIR))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--nsd_tolerance", type=float, default=3.0,
                        help="Surface distance tolerance in pixels for NSD (default 3)")
    args = parser.parse_args()

    masks_dir = Path(args.masks_dir)
    meta_path = masks_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found — run generate_masks.py first")

    with open(meta_path) as f:
        metadata = json.load(f)

    per: dict[str, dict[int, list[float]]] = {k: defaultdict(list)
        for k in ("iou", "dice", "hd95", "sens", "prec", "rve", "nsd")}

    for entry in metadata:
        pseudo_mask = np.load(entry["mask_path"])
        gt_mask     = np.load(entry["gt_mask_path"])
        pred = pseudo_mask >= args.threshold
        gt   = gt_mask > 0.5
        c    = entry["true_label"]

        for key, fn in [
            ("iou",  lambda p, g: _compute_iou(p, g)),
            ("dice", lambda p, g: _compute_dice(p, g)),
            ("hd95", lambda p, g: _compute_hd95(p, g)),
            ("sens", lambda p, g: _compute_sensitivity(p, g)),
            ("prec", lambda p, g: _compute_precision(p, g)),
            ("rve",  lambda p, g: _compute_rve(p, g)),
            ("nsd",  lambda p, g: _compute_nsd(p, g, args.nsd_tolerance)),
        ]:
            val = fn(pred, gt)
            if val is not None:
                per[key][c].append(val)

    header = (f"{'Class':<15} {'N':>5}  {'IoU':>6}  {'Dice':>6}  {'HD95':>7}"
              f"  {'Sens':>6}  {'Prec':>6}  {'RVE':>6}  {'NSD':>6}")
    print(f"\n{header}")
    print("-" * len(header))

    results = {}
    agg: dict[str, list[float]] = {k: [] for k in per}

    for class_idx in range(len(CLASS_NAMES)):
        name = CLASS_NAMES[class_idx]
        n    = len(per["iou"].get(class_idx, []))
        vals = {k: (float(np.mean(per[k][class_idx])) if per[k][class_idx] else float("nan"))
                for k in per}
        results[name] = {"n": n, **vals}
        for k in per:
            agg[k].extend(per[k][class_idx])
        print(f"{name:<15} {n:>5}  {vals['iou']:>6.4f}  {vals['dice']:>6.4f}"
              f"  {vals['hd95']:>7.2f}  {vals['sens']:>6.4f}  {vals['prec']:>6.4f}"
              f"  {vals['rve']:>6.4f}  {vals['nsd']:>6.4f}")

    print("-" * len(header))
    means = {k: float(np.mean(agg[k])) if agg[k] else float("nan") for k in per}
    results["mean"] = means
    print(f"{'mean':<15} {'':>5}  {means['iou']:>6.4f}  {means['dice']:>6.4f}"
          f"  {means['hd95']:>7.2f}  {means['sens']:>6.4f}  {means['prec']:>6.4f}"
          f"  {means['rve']:>6.4f}  {means['nsd']:>6.4f}")

    out_path = masks_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
