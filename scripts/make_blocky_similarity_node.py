"""Blocky 14x14 similarity map for the pipeline figure.

node8 is the *upsampled* 224 map, which looks near-identical to the "Raw Cosine
Similarity Map" panel beside it, so the pair shows no change. This writes the
true pre-upsample view: the 196 cell values the loss actually sees, rendered
with hard edges and the same grid styling node5 uses.

The array is captured by intercepting F.interpolate inside the similarity
engine, so it is the genuine (14, 14) tensor -- not the 224 map pooled back
down.

Usage:
    python scripts/make_blocky_similarity_node.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import src.explainability.patch_similarity as ps            # noqa: E402
from scripts.make_pipeline_node_images import OUT, QUAL_CASES  # noqa: E402
from scripts.make_qualitative_figure import render_case      # noqa: E402
from src.tumor.organ_volume import load_organcam             # noqa: E402
import scripts.compute_index as ci                           # noqa: E402


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, tf = load_organcam(ci.DEFAULT_CKPT, "cuda")

    # Grab the (1,1,14,14) tensor on its way into the bilinear upsample.
    grabbed = {}
    original = ps.F.interpolate

    def spy(t, *a, **kw):
        if hasattr(t, "ndim") and t.ndim == 4 and t.shape[-1] == 14 and t.shape[-2] == 14:
            grabbed["cells"] = t.detach().squeeze().cpu().numpy().copy()
        return original(t, *a, **kw)

    ps.F.interpolate = spy
    try:
        info = QUAL_CASES[0]
        print(f"[blocky] {info['case']}:{info['level']} ({info['top_organ']}) ...", flush=True)
        render_case(info, model, tf, "cuda")
    finally:
        ps.F.interpolate = original

    if "cells" not in grabbed:
        raise SystemExit("did not intercept a 14x14 tensor -- engine path changed?")
    cells = grabbed["cells"]
    assert cells.shape == (14, 14), cells.shape

    # Nearest-neighbour blow-up: each of the 196 cells becomes a hard square,
    # exactly as node5 renders the pooled GT mask.
    blocky = np.kron(cells, np.ones((16, 16)))

    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.imshow(blocky, cmap="viridis")
    for k in range(1, 14):                    # 13 internal lines each way
        ax.axhline(k * 16, color="#888888", linewidth=0.3)
        ax.axvline(k * 16, color="#888888", linewidth=0.3)
    ax.axis("off")
    out = OUT / "node8b_similarity_blocky.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)
    print(f"[blocky] wrote {out}")
    print(f"[blocky] TRUE pre-upsample cells: {cells.min():.4f} .. {cells.max():.4f}")


if __name__ == "__main__":
    main()
