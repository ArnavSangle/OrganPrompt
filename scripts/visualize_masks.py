"""Visualize patch-similarity pseudo-masks overlaid on TotalSegmentator images.

Usage:
    python scripts/visualize_masks.py --root Totalsegmentator_dataset_v201
    python scripts/visualize_masks.py --root Totalsegmentator_dataset_v201 --n_per_class 6
    python scripts/visualize_masks.py --root Totalsegmentator_dataset_v201 --threshold 0.6
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.totalsegmentator import load_totalsegmentator
from src.labels.templates import CLASS_NAMES

# Both live under data/TotalSeg/, not the repo root the old defaults assumed.
DATA_ROOT       = Path("data/TotalSeg/Totalsegmentator_dataset_v201")
MASKS_DIR       = Path("data/TotalSeg/totalseg_masks")
FIGURES_DIR     = Path("figures/totalseg")
THRESHOLDS_JSON = Path("diagnostics/tuned_thresholds.json")

from scripts._figstyle import apply_paper_style

apply_paper_style(font_size=16)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


def get_dataset(root: str | Path = "Totalsegmentator_dataset_v201", top_k: int = 5):
    """Return the TotalSegmentator test split.

    ---- SWAP POINT ----
    Replace body when moving to a new dataset.
    Each item must yield (image_tensor (3, H, W), label_int, gt_mask (224, 224)).
    """
    _, _, test_ds = load_totalsegmentator(root, top_k=top_k)
    return test_ds


def threshold_mask(mask: np.ndarray, percentile: float) -> np.ndarray:
    """Zero out activations below `percentile`, re-normalize to [0, 1]."""
    if percentile <= 0:
        return mask
    cutoff = np.percentile(mask, percentile)
    mask = np.where(mask >= cutoff, mask - cutoff, 0.0)
    peak = mask.max()
    return (mask / peak).astype(np.float32) if peak > 0 else mask


def to_display(image_tensor) -> np.ndarray:
    """Convert (3, H, W) tensor to (H, W, 3) float [0, 1]."""
    img = image_tensor.permute(1, 2, 0).numpy()
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-8)


def overlay(image_np: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend grayscale image with jet-colored heatmap."""
    heatmap = cm.jet(mask)[..., :3]
    return np.clip((1 - alpha) * image_np + alpha * heatmap, 0, 1)


def _load_tuned_thresholds() -> dict[str, float]:
    if not THRESHOLDS_JSON.exists():
        return {}
    with open(THRESHOLDS_JSON) as f:
        data = json.load(f)
    out = {}
    for name, v in data.items():
        if isinstance(v, dict) and "raw" in v:
            out[name] = v["raw"].get("iou_optimal", v["raw"].get("otsu"))
        elif isinstance(v, (int, float)):
            out[name] = float(v)
    return out


def _pick_median_iou_entry(entries: list, threshold: float) -> dict | None:
    """Pick the entry with median IoU (typical case, not cherry-picked)."""
    if not entries:
        return None
    scored = []
    for e in entries:
        mask = np.load(e["mask_path"])
        gt   = np.load(e["gt_mask_path"]) > 0.5
        if not gt.any():
            continue
        pred  = mask >= threshold
        inter = (pred & gt).sum()
        union = (pred | gt).sum()
        iou   = float(inter) / float(union) if union > 0 else 0.0
        scored.append((iou, e))
    if not scored:
        return entries[0]
    scored.sort(key=lambda x: x[0])
    return scored[len(scored) // 2][1]


def plot_class_grid(class_idx: int, class_name: str, entries: list, test_ds, n: int,
                    threshold: float = 0.0):
    """2-row grid: original (top) + overlay (bottom) for n examples of one class."""
    entries = entries[:n]
    if not entries:
        return

    fig, axes = plt.subplots(2, len(entries), figsize=(3.5 * len(entries), 7))
    if len(entries) == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(f"Patch Similarity  ·  {class_name}", fontsize=23, y=1.01)

    for col, entry in enumerate(entries):
        image = test_ds[entry["idx"]][0]   # (3, H, W) — ignore label and gt_mask
        img_np = to_display(image)
        mask   = np.load(entry["mask_path"])

        mask = threshold_mask(mask, threshold)

        axes[0, col].imshow(img_np, cmap="gray")
        axes[0, col].axis("off")
        axes[0, col].set_title(f"#{entry['idx']}", fontsize=12)

        axes[1, col].imshow(overlay(img_np, mask))
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=16)
    axes[1, 0].set_ylabel("Patch Sim", fontsize=16)

    fig.tight_layout()
    safe_name = class_name.replace("-", "_").replace(" ", "_").lower()
    out = FIGURES_DIR / f"totalseg_{safe_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_summary(by_class: dict, test_ds, threshold: float = 0.0):
    """Summary grid: original / mask heatmap / overlay, one column per organ.

    Two stacked blocks of six and five columns, drawn at the printed width
    (\\textwidth = 6.5in). The previous single row of eleven columns was 25.4in
    across and LaTeX shrank it 0.26x on the page, so an 18pt label printed near
    4.6pt. Widening the font could not fix that: eleven columns across 6.5in
    leaves 0.59in each, too narrow for "kidney-right" at any readable size.
    Six columns give ~1in per panel, so the sizes below are literal points on
    the page and the panels themselves are nearly twice as large.
    """
    n_classes = len(CLASS_NAMES)

    # Drawn at the printed width (\textwidth = 6.5in) rather than 25.4in. The
    # old size was shrunk 0.26x by LaTeX, so an 18pt label printed near 4.6pt;
    # at 1:1 the sizes below are literal points on the page. Eleven columns
    # across 6.5in leave 43pt each, and a 12-character name like
    # "kidney-right" needs about 6x its point size, which caps the column
    # titles near 7pt. Going larger than that means rotating them or splitting
    # the grid into two blocks -- it is not reachable in one row.
    # Colorbar lies flat under the grid rather than standing on the right. It
    # reads the same either way, and on the right it cost about half an inch of
    # width -- the dimension that sets the column pitch and therefore caps the
    # organ-label size. Panels sit nearly flush; the gaps were spending more
    # width on nothing.
    fig = plt.figure(figsize=(6.5, 2.5))
    gs  = fig.add_gridspec(
        5, n_classes,
        height_ratios=[1, 1, 1, 0.12, 0.10],
        wspace=0.02, hspace=0.03,
    )
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_classes)]
                     for r in range(3)])
    cbar_ax = fig.add_subplot(gs[4, :])

    tuned = _load_tuned_thresholds()

    mask_im = None
    for class_idx in range(n_classes):
        entries    = by_class.get(class_idx, [])
        class_name = CLASS_NAMES[class_idx]

        for row in range(3):
            axes[row, class_idx].set_xticks([])
            axes[row, class_idx].set_yticks([])
            for spine in axes[row, class_idx].spines.values():
                spine.set_visible(False)

        if not entries:
            continue

        # Pick the median-IoU entry so the figure shows a typical result,
        # not just whatever comes first in the metadata order.
        thr    = tuned.get(class_name, 0.0)
        entry  = _pick_median_iou_entry(entries, thr) or entries[0]
        image  = test_ds[entry["idx"]][0]
        img_np = to_display(image)
        mask   = threshold_mask(np.load(entry["mask_path"]), threshold)

        axes[0, class_idx].imshow(img_np, cmap="gray")
        axes[1, class_idx].imshow(mask, cmap="jet", vmin=0, vmax=1)
        mask_im = axes[1, class_idx].images[0]
        axes[2, class_idx].imshow(overlay(img_np, mask))

        # Capped by column pitch, not by taste: "kidney-right" needs roughly 6x
        # its point size and the pitch is ~39pt once the row labels are paid
        # for, so the kidney pair collides at 7pt even with the colorbar moved
        # off the right edge. Beyond 6.5 the labels have to rotate, or the grid
        # has to split into two blocks of six and five.
        axes[0, class_idx].set_title(class_name, fontsize=6.5, pad=2)

    for row, label in enumerate(["Input CT", "Patch\nSimilarity", "Overlay"]):
        axes[row, 0].set_ylabel(label, fontsize=7, rotation=0,
                                ha="right", va="center", labelpad=8,
                                fontweight="bold")

    if mask_im is not None:
        cbar = fig.colorbar(mask_im, cax=cbar_ax, orientation="horizontal")
        # Not width-constrained the way the column titles are: the bar spans
        # the full figure, so these can be set for legibility alone.
        cbar.set_label("Similarity (post-threshold, normalized)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    # No suptitle: the paper crops this to the image alone and carries the
    # checkpoint name and the one-slice-per-class caveat in the LaTeX caption,
    # so a baked-in title would duplicate it (and set it in the wrong font).
    out = FIGURES_DIR / "totalseg_summary.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DATA_ROOT),
                        help="Path to TotalSegmentator data root")
    parser.add_argument("--masks_dir", default=str(MASKS_DIR),
                        help="Directory of generated masks + metadata.json")
    parser.add_argument("--n_per_class", type=int, default=4,
                        help="Number of examples per class in per-class grids")
    parser.add_argument("--threshold", type=float, default=60.0,
                        help="Zero activations below this percentile (0=off, default=60)")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    masks_dir = Path(args.masks_dir)
    meta_path = masks_dir / "metadata.json"

    def _rebase(stored: str) -> str:
        """Resolve a metadata path against the masks directory actually in use.

        metadata.json records the paths that were current when the masks were
        generated (data/totalseg_masks/...), and the tree has since moved under
        data/TotalSeg/. The layout below masks_dir is unchanged, so rebasing on
        the recorded parent directory name is enough, and an already-correct
        path is returned untouched.
        """
        p = Path(stored)
        if p.exists():
            return str(p)
        # keep any subdirectory (e.g. gt_masks/) that sits below the masks root
        parts = p.parts
        if "totalseg_masks" in parts:
            tail = parts[parts.index("totalseg_masks") + 1:]
            return str(masks_dir.joinpath(*tail))
        return str(masks_dir / p.name)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found — run scripts/generate_masks.py first"
        )

    with open(meta_path) as f:
        metadata = json.load(f)
    for e in metadata:
        e["mask_path"] = _rebase(e["mask_path"])
        e["gt_mask_path"] = _rebase(e["gt_mask_path"])

    print(f"Loaded {len(metadata)} mask records.")
    print("Loading dataset...")
    test_ds = get_dataset(args.root)

    by_class: dict[int, list] = {i: [] for i in range(len(CLASS_NAMES))}
    for entry in metadata:
        by_class[entry["true_label"]].append(entry)

    print(f"Threshold: {args.threshold}th percentile")
    print("Generating per-class grids...")
    for class_idx, class_name in CLASS_NAMES.items():
        plot_class_grid(class_idx, class_name, by_class[class_idx],
                        test_ds, n=args.n_per_class, threshold=args.threshold)

    print("Generating summary figure...")
    plot_summary(by_class, test_ds, threshold=args.threshold)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
