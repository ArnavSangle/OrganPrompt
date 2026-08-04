"""Publication-quality figures for the Phase 2 (bone-metastasis index) writeup.

Two figures, both pure data -- no GPU, no model, no patient volumes:

  figures/paper/fig1_per_level.{svg,png}    per-vertebral-level mean rank vs. chance
  figures/paper/fig2_index_dist.{svg,png}  index-score distribution across the cohort

Reads results/lesion_validation/summary.json and results/bone_index/summary.json,
both already committed. Re-run any time either result changes -- these are cheap
to regenerate and should never go stale relative to what's actually in the repo.

Usage:
    python scripts/make_lesion_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.analyze_lesions import by_level
from src.data.scan_extent import level_rank

OUT = Path(__file__).parent.parent / "figures" / "paper"

# Colorblind-safe (Okabe-Ito): observed = blue, chance = grey, thoracolumbar
# junction (the failure band) highlighted in vermillion.
BLUE = "#0072B2"
GREY = "#999999"
VERMILION = "#D55E00"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "svg.fonttype": "none",
})


def fig1_per_level():
    summary = json.loads(Path("results/lesion_validation/summary.json").read_text())
    lv = by_level(summary["per_lesion"])
    levels = sorted(lv, key=level_rank)  # craniocaudal (T1..T12, L1..L5), not alphabetical
    ranks = [lv[l]["mean_rank"] for l in levels]
    chance = [lv[l]["chance"] for l in levels]
    ns = [lv[l]["n"] for l in levels]

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(levels))
    # Derived from the data, never hardcoded: a hardcoded band silently went stale
    # when the T11/T12 anatomy fix moved T11 from below chance to above it.
    failing = [r >= c for r, c in zip(ranks, chance)]
    colors = [VERMILION if f else BLUE for f in failing]

    ax.bar(x, ranks, width=0.6, color=colors, zorder=3)
    ax.plot(x, chance, marker="o", markersize=3, linewidth=1.0, color=GREY,
             linestyle="--", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=8)
    ax.set_ylabel("mean rank (lower = better)")
    ax.set_ylim(0, max(max(ranks), max(chance)) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3)

    for xi, n in zip(x, ns):
        ax.annotate(f"n={n}", (xi, 0.05), rotation=90, fontsize=6,
                    color="white", ha="center", va="bottom")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    beats = ", ".join(l for l, f in zip(levels, failing) if not f)
    fails = ", ".join(l for l, f in zip(levels, failing) if f)
    handles = [
        Patch(facecolor=BLUE, label=f"beats chance ({beats})"),
        Patch(facecolor=VERMILION, label=f"at/below chance ({fails})"),
        Line2D([0], [0], color=GREY, linestyle="--", marker="o", markersize=3,
               label="chance (order statistic)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT / "fig1_per_level.svg", bbox_inches="tight", transparent=True)
    fig.savefig(OUT / "fig1_per_level.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2_index_dist():
    summary = json.loads(Path("results/bone_index/summary.json").read_text())
    values = np.array(sorted(summary["per_case"].values()))

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.hist(values, bins=12, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
    ax.axvline(summary["mean"], color=VERMILION, linewidth=1.2, linestyle="-",
               zorder=4, label=f"mean = {summary['mean']:.3f}")
    ax.axvline(summary["median"], color=GREY, linewidth=1.2, linestyle="--",
               zorder=4, label=f"median = {summary['median']:.3f}")

    ax.set_xlabel("index score (max blended overlap+proximity)")
    ax.set_ylabel("patients (n)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title(f"n = {summary['n']} patients", fontsize=8, loc="left", color="#555555")

    fig.tight_layout()
    fig.savefig(OUT / "fig2_index_dist.svg", bbox_inches="tight", transparent=True)
    fig.savefig(OUT / "fig2_index_dist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_per_level()
    print(f"[make_lesion_figures] wrote {OUT / 'fig1_per_level.png'}")
    fig2_index_dist()
    print(f"[make_lesion_figures] wrote {OUT / 'fig2_index_dist.png'}")


if __name__ == "__main__":
    main()
