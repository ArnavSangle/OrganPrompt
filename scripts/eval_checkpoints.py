"""Evaluate all segmentation metrics for every checkpoint in a directory.

Loads the model once and swaps weights per checkpoint to avoid GPU memory
churn. Re-encodes text features after each weight swap since the text
encoder is also fine-tuned.

Metrics: IoU, HD95, Dice, Sensitivity, Precision, RVE, NSD

Usage:
    python scripts/eval_checkpoints.py --checkpoint_dir checkpoints/totalseg_patch_v4 --root Totalsegmentator_dataset_v201
    python scripts/eval_checkpoints.py --checkpoint_dir checkpoints/totalseg_patch_v4 --root Totalsegmentator_dataset_v201 --limit 500
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from open_clip import get_tokenizer
from scipy.ndimage import binary_erosion, distance_transform_edt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.totalsegmentator import load_totalsegmentator
from src.labels.templates import CLASS_NAMES, FEW_SHOT_TEMPLATES
from src.explainability.patch_similarity import BioMedCLIPPatchSimilarity

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


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
    pred_within = (distance_transform_edt(~gt_surface)[pred_surface] <= tolerance).sum()
    gt_within   = (distance_transform_edt(~pred_surface)[gt_surface] <= tolerance).sum()
    denom = pred_surface.sum() + gt_surface.sum()
    return float(pred_within + gt_within) / float(denom) if denom > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--limit", type=int, default=500,
                        help="Test images per checkpoint (default 500; use 0 for all)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--nsd_tolerance", type=float, default=3.0,
                        help="Surface distance tolerance in pixels for NSD (default 3)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt_dir  = Path(args.checkpoint_dir)
    ckpt_list = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not ckpt_list:
        print(f"No checkpoints found in {ckpt_dir}")
        return
    print(f"Found {len(ckpt_list)} checkpoints in {ckpt_dir}")

    print("Loading test split...")
    _, _, test_ds = load_totalsegmentator(args.root)
    n = len(test_ds) if args.limit == 0 else min(len(test_ds), args.limit)
    print(f"Evaluating on {n} images per checkpoint\n")

    print("Loading base model...")
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)
    model = model.to(args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer     = get_tokenizer(MODEL_NAME)
    pos_templates = [FEW_SHOT_TEMPLATES[i] for i in range(len(CLASS_NAMES))]
    patch_sim     = BioMedCLIPPatchSimilarity(model, device=args.device)

    metric_keys = ("iou", "dice", "hd95", "sens", "prec", "rve", "nsd")

    header = (f"{'Checkpoint':<45} {'mIoU':>6}  {'Dice':>6}  {'HD95':>7}"
              f"  {'Sens':>6}  {'Prec':>6}  {'RVE':>6}  {'NSD':>6}")
    print(header)
    print("-" * len(header))

    results: dict[str, dict] = {}

    for ckpt_path in ckpt_list:
        ckpt = torch.load(str(ckpt_path), map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()

        tokens = tokenizer(pos_templates).to(args.device)
        with torch.no_grad():
            text_features = F.normalize(model.encode_text(tokens), dim=-1)

        per: dict[str, dict[int, list[float]]] = {k: defaultdict(list) for k in metric_keys}

        for idx in range(n):
            image, label, gt_mask = test_ds[idx]
            image_tensor = image.unsqueeze(0).to(args.device)
            mask = patch_sim.generate(image_tensor, text_features, target_class=label)
            pred = mask >= args.threshold
            gt   = np.asarray(gt_mask) > 0.5

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
                    per[key][label].append(val)

        means = {}
        for key in metric_keys:
            all_vals = [v for vals in per[key].values() for v in vals]
            means[key] = float(np.mean(all_vals)) if all_vals else float("nan")

        results[ckpt_path.name] = means
        print(
            f"{ckpt_path.name:<45} {means['iou']:>6.4f}  {means['dice']:>6.4f}"
            f"  {means['hd95']:>7.2f}  {means['sens']:>6.4f}  {means['prec']:>6.4f}"
            f"  {means['rve']:>6.4f}  {means['nsd']:>6.4f}",
            flush=True,
        )

    patch_sim.remove_hooks()

    best = max(results, key=lambda k: results[k]["iou"])
    print("-" * len(header))
    m = results[best]
    print(
        f"Best: {best}\n"
        f"  mIoU={m['iou']:.4f}  Dice={m['dice']:.4f}  HD95={m['hd95']:.2f}"
        f"  Sens={m['sens']:.4f}  Prec={m['prec']:.4f}  RVE={m['rve']:.4f}  NSD={m['nsd']:.4f}"
    )

    out = ckpt_dir / "checkpoint_metrics.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
