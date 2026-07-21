"""Plot per-class accuracy from eval results.

Loads results/eval_results.json and produces a horizontal bar chart sorted by
accuracy, with a dashed vertical line at overall accuracy.

Usage:
    python scripts/plot_eval.py
    python scripts/plot_eval.py --results results/eval_results.json

Expected JSON structure:
{
    "overall_accuracy": 0.85,
    "per_class_accuracy": {"Bladder": 0.9, "Femur-Left": 0.8, ...},
    "checkpoint": "epoch_009_val3.0132.pt"
}

Output:
    figures/per_class_accuracy.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "results" / "eval_results.json"
FIG_DIR = ROOT / "figures"

FONT = {"family": "sans-serif", "size": 11}


def _apply_base_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(colors="#444444", labelsize=10)
    ax.yaxis.label.set_color("#444444")
    ax.xaxis.label.set_color("#444444")
    ax.title.set_color("#222222")
    ax.grid(False)


def plot_per_class_accuracy(results_path: Path, out_path: Path) -> None:
    """Load eval results and save per-class accuracy bar chart."""
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}\n"
            "Run evaluation first to generate eval_results.json."
        )

    with open(results_path) as f:
        data = json.load(f)

    overall_acc: float = data["overall_accuracy"]
    per_class: dict[str, float] = data["per_class_accuracy"]
    checkpoint: str = data.get("checkpoint", "unknown")

    # Sort ascending so highest bar is at the top when plotted horizontally
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1])
    class_names = [c for c, _ in sorted_classes]
    accuracies = [a for _, a in sorted_classes]

    n = len(class_names)
    fig_height = max(4.0, 0.45 * n + 1.2)
    fig, ax = plt.subplots(figsize=(7.5, fig_height), dpi=150)
    matplotlib.rcParams.update({"font.family": "sans-serif"})

    # Colour bars by performance: below overall → muted, above → accented
    bar_colors = [
        "#2563EB" if acc >= overall_acc else "#94A3B8"
        for acc in accuracies
    ]

    bars = ax.barh(
        class_names,
        accuracies,
        color=bar_colors,
        height=0.6,
        zorder=2,
    )

    # Overall accuracy dashed vertical line
    ax.axvline(
        overall_acc,
        color="#EF4444",
        linewidth=1.5,
        linestyle="--",
        zorder=3,
        label=f"Overall  {overall_acc:.1%}",
    )

    # Value labels on each bar
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.1%}",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#444444",
        )

    ax.set_xlabel("Accuracy", fontdict=FONT)
    ax.set_title(
        f"Per-Class Accuracy — {checkpoint}",
        fontdict={**FONT, "size": 13, "weight": "semibold"},
        pad=12,
    )
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_xlim(0, min(1.0, max(accuracies) + 0.12))
    _apply_base_style(ax)

    ax.legend(
        frameon=False,
        fontsize=9,
        loc="lower right",
        labelcolor="#444444",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"Path to eval_results.json (default: {DEFAULT_RESULTS})",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=FIG_DIR / "per_class_accuracy.png",
        help="Output figure path",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading results from {args.results} …")
    plot_per_class_accuracy(args.results, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
