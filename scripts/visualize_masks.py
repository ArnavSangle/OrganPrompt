"""Visualise Grad-CAM pseudo-masks overlaid on OrganAMNIST images.

Usage:
    python scripts/visualize_masks.py
    python scripts/visualize_masks.py --n_per_class 6
    python scripts/visualize_masks.py --threshold 0.6   # zero below 60th percentile
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.organmnist import load_organamnist
from src.labels.templates import CLASS_NAMES

MASKS_DIR   = Path("data/pseudo_masks")
FIGURES_DIR = Path("figures/gradcam")

plt.rcParams.update({"font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})


def threshold_mask(mask: np.ndarray, percentile: float) -> np.ndarray:
    """Zero out activations below `percentile`, re-normalise to [0, 1]."""
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
    """Blend grayscale image with jet-coloured heatmap."""
    heatmap = cm.jet(mask)[..., :3]
    return np.clip((1 - alpha) * image_np + alpha * heatmap, 0, 1)


def plot_class_grid(class_idx: int, class_name: str, entries: list, test_ds, n: int,
                    threshold: float = 0.0):
    """2-row grid: original (top) + overlay (bottom) for n examples of one class."""
    entries = entries[:n]
    if not entries:
        return

    fig, axes = plt.subplots(2, len(entries), figsize=(3.5 * len(entries), 7))
    if len(entries) == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(f"Grad-CAM  ·  {class_name}", fontsize=13, y=1.01)

    for col, entry in enumerate(entries):
        image, _ = test_ds[entry["idx"]]
        img_np   = to_display(image)
        mask     = np.load(entry["mask_path"])

        mask = threshold_mask(mask, threshold)

        axes[0, col].imshow(img_np, cmap="gray")
        axes[0, col].axis("off")
        axes[0, col].set_title(f"#{entry['idx']}", fontsize=7)

        axes[1, col].imshow(overlay(img_np, mask))
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("GradCAM", fontsize=9)

    fig.tight_layout()
    safe_name = class_name.replace("-", "_").replace(" ", "_").lower()
    out = FIGURES_DIR / f"gradcam_{safe_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_summary(by_class: dict, test_ds, threshold: float = 0.0):
    """3-row × 11-col summary: original / mask heatmap / overlay."""
    n_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(3, n_classes, figsize=(3 * n_classes, 9))

    for class_idx in range(n_classes):
        entries    = by_class.get(class_idx, [])
        class_name = CLASS_NAMES[class_idx]

        if not entries:
            for row in range(3):
                axes[row, class_idx].axis("off")
            continue

        entry    = entries[0]
        image, _ = test_ds[entry["idx"]]
        img_np   = to_display(image)
        mask     = np.load(entry["mask_path"])

        mask = threshold_mask(mask, threshold)

        axes[0, class_idx].imshow(img_np, cmap="gray")
        axes[0, class_idx].set_title(class_name, fontsize=7)
        axes[0, class_idx].axis("off")

        axes[1, class_idx].imshow(mask, cmap="jet", vmin=0, vmax=1)
        axes[1, class_idx].axis("off")

        axes[2, class_idx].imshow(overlay(img_np, mask))
        axes[2, class_idx].axis("off")

    for row, label in enumerate(["Original", "Mask", "Overlay"]):
        axes[row, 0].set_ylabel(label, fontsize=9)

    fig.suptitle("Grad-CAM Summary — All 11 Organs", fontsize=14, y=1.01)
    fig.tight_layout()
    out = FIGURES_DIR / "gradcam_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_class", type=int, default=4,
                        help="Number of examples per class in per-class grids")
    parser.add_argument("--threshold", type=float, default=60.0,
                        help="Zero activations below this percentile (0=off, default=60)")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    meta_path = MASKS_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found — run scripts/generate_masks.py first"
        )

    with open(meta_path) as f:
        metadata = json.load(f)

    print(f"Loaded {len(metadata)} mask records.")
    print("Loading OrganAMNIST test split...")
    _, _, test_ds = load_organamnist(size=224)

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
