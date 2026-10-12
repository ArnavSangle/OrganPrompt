"""Supplementary Figure S1: the model along each organ's craniocaudal extent.

Reviewer 3 (Q5) asked whether the model generalizes across anatomical locations.
Every evaluation slice is one of the five largest cross-sections of its organ, so
Figure 2 and the metrics only show organ interiors. This figure takes one test
volume per class (seeded choice among test volumes containing the organ) and
shows the slices at the 10th, 50th and 90th percentile of that organ's own
z-extent: near both ends and at the middle.

Each panel: CT, the patch-similarity map thresholded with the validation-tuned
threshold for that class (results/equal_supervision/organprompt.json, written by
`linear_probe.py eval --method organprompt`), and the ground-truth outline, with
the slice's IoU. Needs the model: run it yourself after the eval job.

    python scripts/make_organ_extent_figure.py
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_erosion

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._figstyle import apply_paper_style
from src.labels.templates import CLASS_NAMES
from src.data.totalsegmentator import (ORGAN_FILES, TotalSegmentatorDataset,
                                       load_totalsegmentator)

PCTS = (10, 50, 90)
BLUE, BRICK = "#3B6EA5", "#B4543A"


def extent_slices(coverage, pcts=PCTS):
    """Slice indices at the given percentiles of the organ's own z-range,
    snapped to the nearest slice that actually contains the organ."""
    zs = np.flatnonzero(np.asarray(coverage) > 0)
    lo, hi = zs.min(), zs.max()
    out = []
    for p in pcts:
        target = lo + (hi - lo) * p / 100.0
        out.append(int(zs[np.argmin(np.abs(zs - target))]))
    return out


def _coverage(vol_dir, organ_class):
    merged = None
    for name in ORGAN_FILES[organ_class]:
        d = np.asarray(nib.load(Path(vol_dir) / "segmentations" / f"{name}.nii.gz").dataobj) > 0
        merged = d if merged is None else (merged | d)
    return merged.sum(axis=(0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/TotalSeg/Totalsegmentator_dataset_v201")
    ap.add_argument("--checkpoint",
                    default="checkpoints/totalseg_patch_v5/epoch_011_val3.4584.pt")
    ap.add_argument("--thresholds", default="results/equal_supervision/organprompt.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="paper/figures_for_upload/organ_extent_S1.png")
    a = ap.parse_args()

    from scripts.linear_probe import _backbone, _organprompt_scorer
    thr = json.loads(Path(a.thresholds).read_text())["thresholds_val_tuned"]
    _, _, test_ds = load_totalsegmentator(a.root)
    vols_by_class = {}
    for it in test_ds.items:
        vols_by_class.setdefault(it["organ_class"], set()).add(it["volume_dir"])
    rng = random.Random(a.seed)
    score = _organprompt_scorer(_backbone(a.device, a.checkpoint), a.device)

    apply_paper_style(font_size=7)
    n = len(CLASS_NAMES)
    fig, axes = plt.subplots(n, len(PCTS), figsize=(4.2, 1.45 * n))
    record = {}
    for c in range(n):
        vol = rng.choice(sorted(vols_by_class[c]))
        zs = extent_slices(_coverage(vol, c))
        items = [{"volume_dir": vol, "slice_idx": z, "organ_class": c} for z in zs]
        ds = TotalSegmentatorDataset(items, eval_mode=True)
        record[CLASS_NAMES[c]] = {"volume": Path(vol).name, "slices": zs, "iou": []}
        for j in range(len(zs)):
            img, _, gt = ds[j]
            sim = score(img, c)
            pred = sim >= thr[CLASS_NAMES[c]]
            g = np.asarray(gt) > 0.5
            u = (pred | g).sum()
            iou = float((pred & g).sum() / u) if u else float("nan")
            record[CLASS_NAMES[c]]["iou"].append(iou)
            ax = axes[c, j]
            ax.imshow(img[0].numpy(), cmap="gray")
            ax.imshow(np.ma.masked_where(~pred, pred), cmap=ListedColormap([BLUE]),
                      alpha=0.55)
            edge = g ^ binary_erosion(g)
            ax.imshow(np.ma.masked_where(~edge, edge), cmap=ListedColormap([BRICK]))
            ax.text(0.03, 0.05, f"IoU {iou:.2f}", color="white", fontsize=6,
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_title(f"{PCTS[j]}th pct of extent", fontsize=7)
        axes[c, 0].set_ylabel(CLASS_NAMES[c], fontsize=7)
    fig.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300)
    side = Path(a.out).with_suffix(".json")
    side.write_text(json.dumps(record, indent=2))
    print(f"wrote {a.out} and {side}")


if __name__ == "__main__":
    main()
