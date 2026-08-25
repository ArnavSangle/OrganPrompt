"""CT slice with the 14x14 patch lattice drawn over it -- the encoder INPUT
node for the pipeline figure.

Two versions, because the two panels run on different cohorts and the figure
should not hide that:

    node0a_input_lattice_totalseg.png   TotalSegmentator slice  (panel A)
    node0b_input_lattice_spinemets.png  Spine-Mets 10355:T3     (panel B)

224 / 16 = 14, so the lattice lines fall every 16 px. Same 0.3pt #888888
styling node5 uses, so all the 14x14 nodes read as one family.

Usage:
    python scripts/make_lattice_input_node.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.make_pipeline_node_images import OUT, QUAL_CASES  # noqa: E402
from scripts.make_qualitative_figure import render_case        # noqa: E402
from src.tumor.organ_volume import load_organcam               # noqa: E402
import scripts.compute_index as ci                             # noqa: E402


def save_lattice(ct_disp, out_name):
    """ct_disp is a float [0,1] slice at native resolution."""
    h, w = ct_disp.shape
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.imshow(ct_disp, cmap="gray", extent=[0, 224, 224, 0])
    for k in range(1, 14):                     # 13 internal lines each way
        ax.axhline(k * 16, color="#888888", linewidth=0.3)
        ax.axvline(k * 16, color="#888888", linewidth=0.3)
    ax.set_xlim(0, 224); ax.set_ylim(224, 0)
    ax.axis("off")
    out = OUT / out_name
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)
    print(f"[lattice] wrote {out}  (source slice {h}x{w})")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, tf = load_organcam(ci.DEFAULT_CKPT, "cuda")

    info = QUAL_CASES[0]
    print(f"[lattice] panel B source: {info['case']}:{info['level']}", flush=True)
    ct_disp, _, _ = render_case(info, model, tf, "cuda")
    save_lattice(ct_disp, "node0b_input_lattice_spinemets.png")


if __name__ == "__main__":
    main()
