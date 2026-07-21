"""Evaluate pseudo-masks against TotalSegmentator ground-truth.

Metrics: IoU, HD95, Dice, Sensitivity, Precision, RVE, NSD

Usage:
    python scripts/evaluate_masks.py
    python scripts/evaluate_masks.py --tune                       # sweep IoU-optimal per class, save, evaluate
    python scripts/evaluate_masks.py --thresholds_json diagnostics/tuned_thresholds.json
    python scripts/evaluate_masks.py --threshold 0.5
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


def _tune_thresholds(metadata, n_candidates=100):
    """Sweep candidate thresholds per class to maximize IoU on the generated masks."""
    per_class_pairs: dict[int, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for entry in metadata:
        m  = np.load(entry["mask_path"])
        gt = np.load(entry["gt_mask_path"]) > 0.5
        per_class_pairs[entry["true_label"]].append((m, gt))

    tuned: dict[int, float] = {}
    print("\nTuning per-class IoU-optimal thresholds:")
    for c, pairs in per_class_pairs.items():
        lo = min(m.min() for m, _ in pairs)
        hi = max(m.max() for m, _ in pairs)
        candidates = np.linspace(lo, hi, n_candidates)
        best_thr, best_iou = float(lo), -1.0
        for thr in candidates:
            ious = []
            for m, gt in pairs:
                if not gt.any():
                    continue
                pred  = m >= thr
                inter = (pred & gt).sum()
                union = (pred | gt).sum()
                if union > 0:
                    ious.append(inter / union)
            if not ious:
                continue
            miou = float(np.mean(ious))
            if miou > best_iou:
                best_iou, best_thr = miou, float(thr)
        tuned[c] = best_thr
        print(f"  {CLASS_NAMES[c]:<15} thr={best_thr:+.4f}  IoU={best_iou:.4f}")
    return tuned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks_dir", default=str(DEFAULT_MASKS_DIR))
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Uniform threshold when --thresholds_json / --tune is not given")
    parser.add_argument("--thresholds_json", default=None,
                        help="Path to tuned_thresholds.json for per-class thresholds")
    parser.add_argument("--tune", action="store_true",
                        help="Sweep IoU-optimal thresholds per class, save to "
                             "diagnostics/tuned_thresholds.json, then evaluate with them")
    parser.add_argument("--nsd_tolerance", type=float, default=3.0,
                        help="Surface distance tolerance in pixels for NSD (default 3)")
    args = parser.parse_args()

    masks_dir = Path(args.masks_dir)
    meta_path = masks_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found — run generate_masks.py first")

    with open(meta_path) as f:
        metadata = json.load(f)

    thresholds: dict[int, float] = {}
    if args.tune:
        thresholds = _tune_thresholds(metadata)
        out = Path("diagnostics") / "tuned_thresholds.json"
        out.parent.mkdir(exist_ok=True)
        with open(out, "w") as f:
            json.dump({CLASS_NAMES[c]: t for c, t in thresholds.items()}, f, indent=2)
        print(f"\nTuned thresholds saved to {out}")
    elif args.thresholds_json:
        with open(args.thresholds_json) as f:
            data = json.load(f)
        for c, name in CLASS_NAMES.items():
            v = data.get(name)
            if isinstance(v, dict) and "raw" in v:
                thresholds[c] = v["raw"].get("iou_optimal", v["raw"].get("otsu"))
            elif isinstance(v, (int, float)):
                thresholds[c] = float(v)
        print(f"Per-class thresholds loaded from {args.thresholds_json}:")
        for c in sorted(thresholds):
            print(f"  {CLASS_NAMES[c]:<15} {thresholds[c]:+.4f}")
    else:
        print(f"Uniform threshold: {args.threshold}")

    per: dict[str, dict[int, list[float]]] = {k: defaultdict(list)
        for k in ("iou", "dice", "hd95", "sens", "prec", "rve", "nsd")}

    for entry in metadata:
        pseudo_mask = np.load(entry["mask_path"])
        gt_mask     = np.load(entry["gt_mask_path"])
        c    = entry["true_label"]
        thr  = thresholds.get(c, args.threshold)
        pred = pseudo_mask >= thr
        gt   = gt_mask > 0.5

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
