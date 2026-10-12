"""Publication-quality figures for the Phase 2 (bone-metastasis index) writeup.

Two figures, both pure data -- no GPU, no model, no patient volumes:

  fig1_per_level  per-vertebral-level mean rank vs. chance
  fig2_rank_dist  rank of the first expected organ over all lesions vs. its exact
                  chance distribution (BSPC revision: replaces the per-patient
                  "index" histogram, since the alpha/beta blend reduces to distance
                  and the patient-level maximum is no longer reported)

Written to figures/paper/ and copied under the flat names the manuscript uses
(paper/figures_for_upload/per_level_rank.png, rank_distribution.png).

Usage:
    python scripts/make_lesion_figures.py --summary results/lesion_validation_v2/summary.json
"""
import argparse
import shutil
from math import comb
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.analyze_lesions import by_level
from src.data.scan_extent import level_rank

OUT = Path(__file__).parent.parent / "figures" / "paper"
UPLOAD = Path(__file__).parent.parent / "paper" / "figures_for_upload"


def chance_rank_pmf(k, n=11):
    """P(best rank of k expected organs == r), r = 1..n, under a random ranking:
    C(n-r, k-1) / C(n, k). Its mean is (n+1)/(k+1), the chance used everywhere."""
    return np.array([comb(n - r, k - 1) / comb(n, k) for r in range(1, n + 1)])


def rank_histogram(per_lesion, n=11):
    """(observed counts per rank, expected counts under chance), ranked lesions only."""
    obs, exp = np.zeros(n), np.zeros(n)
    for v in per_lesion.values():
        if v["rank"] is None:
            continue
        obs[v["rank"] - 1] += 1
        exp += chance_rank_pmf(len(v["expected"]), n)
    return obs, exp

# Two-hue categorical set, muted for print. Validated colorblind-safe: passes
# lightness band, chroma floor, CVD separation (worst adjacent dE 16.2 protan)
# and normal-vision separation (dE 22.6). Do not desaturate further -- muting
# these below the current chroma drops them under the CVD separation floor.
BLUE = "#3B6EA5"       # beats chance
BRICK = "#B4543A"      # at or below chance
# Reference lines are annotation, not a category, so they wear ink rather than a
# hue -- that keeps the two hues above meaning exactly one thing each.
INK = "#1A1A1A"

def _register_latin_modern():
    """Make the paper's body font available to matplotlib.

    main.tex loads lmodern, so figure text should be Latin Modern Roman rather
    than a sans fallback. TeX Live ships the OTF but does not install it as a
    system font, so matplotlib cannot see it until it is registered here. Silent
    no-op when TeX Live is absent -- font.serif then falls through to DejaVu.
    """
    import matplotlib.font_manager as fm
    for root in (Path(r"C:/texlive"), Path("/usr/share/texmf-dist")):
        if not root.exists():
            continue
        for otf in root.glob("**/fonts/opentype/public/lm/lmroman10-*.otf"):
            try:
                fm.fontManager.addfont(str(otf))
            except Exception:
                pass


_register_latin_modern()

plt.rcParams.update({
    # Serif to match the paper body (lmodern). Latin Modern Roman is registered
    # above from TeX Live; DejaVu Serif is the always-present fallback.
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 16,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "svg.fonttype": "none",
})


def fig1_per_level(summary_path):
    summary = json.loads(Path(summary_path).read_text())
    lv = by_level(summary["per_lesion"])
    levels = sorted(lv, key=level_rank)  # craniocaudal (T1..T12, L1..L5), not alphabetical
    ranks = [lv[l]["mean_rank"] for l in levels]
    chance = [lv[l]["chance"] for l in levels]
    ns = [lv[l]["n"] for l in levels]

    # Width matches the printed width (\textwidth = 6.5in at letterpaper/1in
    # margins), so LaTeX does not rescale and the font sizes below are literal
    # points on the page.
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    x = np.arange(len(levels))
    # Derived from the data, never hardcoded: a hardcoded band silently went stale
    # when the T11/T12 anatomy fix moved T11 from below chance to above it.
    failing = [r >= c for r, c in zip(ranks, chance)]
    colors = [BRICK if f else BLUE for f in failing]

    ax.bar(x, ranks, width=0.6, color=colors, zorder=3)
    ax.plot(x, chance, marker="o", markersize=3, linewidth=1.0, color=INK,
             linestyle="--", zorder=4)

    ax.set_xticks(x)
    # Counts ride with the tick labels. Previously they were white rotated text
    # inside each bar, which disappeared off the top of the shortest bars.
    # Bare count on the second line: "(n=17)" at 17 levels across 6.5in collides
    # with its neighbors, so the "n=" is stated once in the axis label instead.
    ax.set_xticklabels([f"{l}\n{n}" for l, n in zip(levels, ns)], fontsize=12)
    ax.set_xlabel("vertebral level (lesion count below each label)", fontsize=14)
    # Explicit, smaller than the 16pt base: rotated 90 degrees, this string is
    # long enough that the base size clipped off the figure's left edge.
    # labelpad lifts it clear of the tick numbers, which it otherwise touches.
    ax.set_ylabel("mean rank (lower = better)", fontsize=13, labelpad=12)
    ax.set_ylim(0, max(max(ranks), max(chance)) * 1.15)
    # Integer ticks: the stacked legend shortens the axes enough that the
    # default locator falls back to 0.0/2.5, which reads oddly for a rank.
    ax.set_yticks(np.arange(0, int(max(max(ranks), max(chance))) + 1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    # Labels stay short and the key sits ABOVE the axes: interpolating the level
    # names produced a legend string wider than the plot, laid over the bars.
    # The bar colors already show which levels fail, so the key only has to say
    # what the colors mean.
    handles = [
        Patch(facecolor=BLUE, label="beats chance"),
        Patch(facecolor=BRICK, label="at or below chance"),
        Line2D([0], [0], color=INK, linestyle="--", marker="o", markersize=3,
               label="chance (order statistic)"),
    ]
    # Stacked one per line rather than spread across the top: the three labels
    # read as a list this way instead of running together into one strip.
    ax.legend(handles=handles, frameon=False, fontsize=12,
              loc="lower left", bbox_to_anchor=(0, 1.02),
              ncol=1, borderaxespad=0, labelspacing=0.35, handlelength=1.8)

    fig.tight_layout()
    fig.savefig(OUT / "fig1_per_level.svg", bbox_inches="tight", transparent=True)
    fig.savefig(OUT / "fig1_per_level.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2_rank_dist(summary_path):
    summary = json.loads(Path(summary_path).read_text())
    obs, exp = rank_histogram(summary["per_lesion"])
    r = np.arange(1, len(obs) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.bar(r, obs, width=0.6, color=BLUE, zorder=3, label="observed")
    ax.plot(r, exp, marker="o", markersize=3, linewidth=1.0, color=INK,
            linestyle="--", zorder=4, label="expected under chance")
    ax.set_xticks(r)
    ax.set_xlabel("rank of the first expected organ", fontsize=13)
    ax.set_ylabel("lesions (n)", fontsize=13, labelpad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(frameon=False, fontsize=12, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_rank_dist.svg", bbox_inches="tight", transparent=True)
    fig.savefig(OUT / "fig2_rank_dist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/lesion_validation_v2/summary.json")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    UPLOAD.mkdir(parents=True, exist_ok=True)
    fig1_per_level(a.summary)
    fig2_rank_dist(a.summary)
    shutil.copy(OUT / "fig1_per_level.png", UPLOAD / "per_level_rank.png")
    shutil.copy(OUT / "fig2_rank_dist.png", UPLOAD / "rank_distribution.png")
    print(f"[make_lesion_figures] wrote {OUT} and copies in {UPLOAD}")


if __name__ == "__main__":
    main()
