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


def fig1_per_level():
    summary = json.loads(Path("results/lesion_validation/summary.json").read_text())
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


def fig2_index_dist():
    summary = json.loads(Path("results/bone_index/summary.json").read_text())
    values = np.array(sorted(summary["per_case"].values()))

    # Was 3.4in wide but placed at \textwidth, so LaTeX blew it up 1.9x and every
    # 9pt label rendered near 17pt. Drawn at printed width instead.
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.hist(values, bins=12, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
    ax.axvline(summary["mean"], color=BRICK, linewidth=1.2, linestyle="-",
               zorder=4, label=f"mean = {summary['mean']:.3f}")
    ax.axvline(summary["median"], color=INK, linewidth=1.2, linestyle="--",
               zorder=4, label=f"median = {summary['median']:.3f}")

    ax.set_xlabel("index score (max blended overlap+proximity)", labelpad=10)
    ax.set_ylabel("patients (n)", labelpad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3)
    # Key sits inside the axes at top right, stacked one entry per line: the
    # distribution is right-skewed, so that corner is empty in every run.
    # The embedded title is gone -- the LaTeX \caption already names the figure,
    # and n is stated there.
    ax.legend(frameon=False, fontsize=12, loc="upper right", ncol=1,
              borderaxespad=0.8, labelspacing=0.4, handlelength=1.8)

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
