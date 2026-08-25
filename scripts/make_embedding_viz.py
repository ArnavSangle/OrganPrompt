"""Visualize the real liver text embedding (node 4's output), for the
hand-built pipeline schematic. Companion to make_pipeline_node_images.py.

Deliberately rendered as a 1D strip, NOT reshaped into a 2D grid -- the
embedding genuinely is a flat 512-number vector (one per prompt, pooled),
and drawing it as a grid would silently reintroduce the exact "text has a
spatial grid too" confusion the node-4 icon redesign specifically corrected.

Usage:
    python scripts/make_embedding_viz.py --device cuda
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts._figstyle import apply_paper_style
from src.tumor.organ_volume import load_organcam
from src.labels.templates import CLASS_NAMES

OUT = Path("figures/paper/pipeline_nodes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/totalseg_patch_v5/epoch_011_val3.4584.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    apply_paper_style()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[embed_viz] loading model + encoding all 11 organ prompts ...")
    model, tf = load_organcam(args.checkpoint, args.device)
    liver_idx = {v: k for k, v in CLASS_NAMES.items()}["liver"]
    vec = tf[liver_idx].detach().cpu().numpy()  # (512,), L2-normalized
    print(f"[embed_viz] liver embedding: {vec.shape[0]}-d, "
          f"||v||={np.linalg.norm(vec):.3f}, range [{vec.min():.3f}, {vec.max():.3f}]")

    fig, ax = plt.subplots(figsize=(5.0, 0.9))
    strip = vec.reshape(1, -1)
    # Scale by std, not by the absolute max: a handful of outlier dimensions
    # (e.g. the 0.632 peak) would otherwise saturate the color range and wash
    # out the rest of the 512 values to near-white. 3*std clips those outliers
    # to full color and gives the bulk of the distribution real contrast.
    vmax = 3 * vec.std()
    print(f"[embed_viz] color scale +/-{vmax:.3f} (3*std={vec.std():.3f}); "
          f"{(np.abs(vec) > vmax).sum()} of 512 values clipped/saturated")
    im = ax.imshow(strip, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks([])
    ax.set_xticks([0, 511])
    ax.set_xticklabels(["1", "512"])
    # No "embedding dimension" xlabel text -- added manually in the diagram
    # instead. Tick labels 1/512 stay, since those are part of the data
    # (dimension indices), not a caption.
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.06, pad=0.02)
    cbar.ax.tick_params(labelsize=6)

    fig.savefig(OUT / "node4_text_embedding.png", dpi=200, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"[embed_viz] wrote {OUT / 'node4_text_embedding.png'}")


if __name__ == "__main__":
    main()
