"""Replacement for Figure 1 (pipeline), panel B, last node (BSPC revision).

The submitted node 11 ("Metastasis Index -- alpha*overlap + beta*proximity",
index = 0.202) drew the involved vertebra with the mirrored, index-for-index SEG
placement and illustrated the blended index, which the revision replaces with a
ranking by distance from the involved vertebra. This renders the same case
(10355 T3) the way Figure 5 does: the assigned mask of the top-ranked organ on
the slice of its voxel nearest the vertebra, the vertebra outline, and the
nearest-pair segment. Same size and dpi as the other pipeline node images.
Figure 1 itself is assembled outside this repository.

    python scripts/make_node11_ranking.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).parent.parent))

OUT = Path("figures/paper/pipeline_nodes/node11_distance_ranking.png")


def panel_label(info):
    return f"{info['top_organ']}, {info['distance_mm']:.0f} mm"


def main():
    from src.tumor.crossfit import load_crossfit
    from src.tumor.organ_volume import load_organcam
    import scripts.compute_index as ci
    from scripts.make_qualitative_figure import case_from_summary, render_case
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/lesion_validation_v2/summary.json")
    ap.add_argument("--case", default="10355")
    ap.add_argument("--level", default="T3")
    a = ap.parse_args()
    info = case_from_summary(json.loads(Path(a.summary).read_text()), a.case, a.level)
    calib = load_crossfit("results/crossfit_folds.json", "results/organ_stats_all55.json")
    model, tf = load_organcam(ci.DEFAULT_CKPT, "cuda")
    r = render_case(info, model, tf, "cuda", calib)

    color = "#3B6EA5" if info["correct"] else "#B4543A"
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.imshow(r["ct"], cmap="gray")
    ax.imshow(np.ma.masked_where(~r["organ"], r["organ"]),
              cmap=ListedColormap([color]), alpha=0.6)
    ax.imshow(np.ma.masked_where(~r["vertebra_outline"], r["vertebra_outline"]),
              cmap=ListedColormap(["#E0301E"]), alpha=0.95)
    (vy, vx), (oy, ox) = r["p_vertebra"], r["p_organ"]
    ax.plot([vx, ox], [vy, oy], color="white", linewidth=0.8)
    ax.plot([vx, ox], [vy, oy], "o", color="white", markersize=1.6)
    ax.text(0.03, 0.04, panel_label(info), color="white", fontsize=6,
            transform=ax.transAxes)
    ax.axis("off")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    print(f"[node11] {info['case']}:{info['level']} {panel_label(info)} "
          f"({r['distance_mm']:.1f} mm recomputed) -> {OUT}")


if __name__ == "__main__":
    main()
