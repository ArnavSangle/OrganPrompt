"""Generate summary figures from committed results.

Two figures:
    figures/progression.png         methodology arc: baseline → v4-tuned
    figures/calibration_impact.png  per-class v4-uniform vs v4-tuned

Data is hardcoded from results already in the repo. Pure plotting — no
model inference — runs in seconds without GPU.

Usage:
    python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("figures")

# ─── Methodology progression (raw untuned checkpoint peaks, unless noted) ────
PROGRESSION = [
    ("Baseline\n(vanilla-gradient CAM)", 0.003),
    ("v0\nGradCAM + SigLIP",             0.063),
    ("v1\n+ patch-level loss",           0.307),
    ("v3\n+ resume from v1",             0.469),
    ("v4\n+ resume from v3",             0.574),
    ("v4-tuned\n+ calibration",          0.633),
    ("v5-tuned\n+ resume from v4",       0.653),
]

CLASSES = ["bladder", "femur-left", "femur-right", "heart",
           "kidney-left", "kidney-right", "liver",
           "lung-left", "lung-right", "pancreas", "spleen"]

# ─── Per-class IoU: v4 with uniform threshold=0.5 vs v5 tuned (final model) ──
V4_UNIFORM = {
    "bladder": 0.663, "femur-left": 0.449, "femur-right": 0.386,
    "heart": 0.763, "kidney-left": 0.263, "kidney-right": 0.265,
    "liver": 0.782, "lung-left": 0.753, "lung-right": 0.765,
    "pancreas": 0.124, "spleen": 0.480,
}
V5_TUNED = {
    "bladder": 0.698, "femur-left": 0.587, "femur-right": 0.565,
    "heart": 0.787, "kidney-left": 0.491, "kidney-right": 0.453,
    "liver": 0.823, "lung-left": 0.798, "lung-right": 0.828,
    "pancreas": 0.290, "spleen": 0.644,
}


def figure_progression():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = [x[0] for x in PROGRESSION]
    values = [x[1] for x in PROGRESSION]

    # gradient from grey → deep blue for the methodology arc
    cmap   = plt.get_cmap("viridis")
    colors = [cmap(i / (len(labels) - 1)) for i in range(len(labels))]
    colors[-2] = "#c1272d"  # highlight the calibration step in red
    colors[-1] = "#8a1b21"  # v5-tuned in darker red (final model)

    bars = ax.bar(range(len(labels)), values, color=colors,
                  edgecolor="black", linewidth=0.7)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("mIoU on TotalSegmentator test set", fontsize=12)
    ax.set_ylim(0, 0.75)
    ax.set_title(
        "OrganCAM methodology progression (baseline → final)",
        fontsize=13,
    )
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

    ax.annotate(
        "no retrain\n+0.059 mIoU",
        xy=(len(labels) - 2, values[-2]),
        xytext=(len(labels) - 2.5, values[-2] + 0.09),
        fontsize=9, ha="center", color="#c1272d",
        arrowprops=dict(arrowstyle="->", color="#c1272d", lw=1.2),
    )
    ax.annotate(
        "final model\n+0.020 mIoU",
        xy=(len(labels) - 1, values[-1]),
        xytext=(len(labels) - 0.6, values[-1] + 0.09),
        fontsize=9, ha="center", color="#8a1b21",
        arrowprops=dict(arrowstyle="->", color="#8a1b21", lw=1.2),
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "progression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_calibration_impact():
    ordered  = sorted(CLASSES, key=lambda c: -V4_UNIFORM[c])
    uniform  = np.array([V4_UNIFORM[c] for c in ordered])
    tuned    = np.array([V5_TUNED[c]   for c in ordered])
    x        = np.arange(len(ordered))
    w        = 0.4

    fig, ax = plt.subplots(figsize=(12.5, 6))
    ax.bar(x - w/2, uniform, w, label="v4 (uniform threshold = 0.5)",
           color="#a5b6d0", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, tuned,   w, label="v5-tuned (final model + per-class thresholds)",
           color="#8a1b21", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=22, ha="right", fontsize=11)
    ax.set_ylabel("IoU", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Per-class gains from calibration + one additional resume run (v4 → v5-tuned)",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # annotate improvement (or lack thereof)
    for i, (u, t) in enumerate(zip(uniform, tuned)):
        delta = t - u
        y = max(u, t) + 0.025
        col = "#0a7a0a" if delta > 0.05 else "#555555"
        ax.text(x[i], y, f"{'+' if delta >= 0 else ''}{delta:.2f}",
                ha="center", fontsize=10, color=col, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "calibration_impact.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_progression()
    print(f"Saved {OUT_DIR / 'progression.png'}")
    figure_calibration_impact()
    print(f"Saved {OUT_DIR / 'calibration_impact.png'}")


if __name__ == "__main__":
    main()
