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
# Short, single-line labels so they don't overlap at the given figure width.
PROGRESSION = [
    ("baseline",       0.003),
    ("v0",             0.063),
    ("v1",             0.307),
    ("v3",             0.469),
    ("v4",             0.574),
    ("v4-tuned",       0.633),
    ("v5-tuned",       0.653),
]
PROGRESSION_DESCRIPTIONS = [
    "vanilla-gradient CAM",
    "GradCAM + SigLIP fine-tuning",
    "+ patch-level contrastive loss",
    "+ resume from v1",
    "+ resume from v3",
    "+ inference calibration",
    "+ resume from v4",
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

    cmap   = plt.get_cmap("viridis")
    colors = [cmap(i / (len(labels) - 1)) for i in range(len(labels))]
    colors[-2] = "#c1272d"
    colors[-1] = "#8a1b21"

    bars = ax.bar(range(len(labels)), values, color=colors,
                  edgecolor="black", linewidth=0.7)

    # Version labels on the tick axis (short, no overlap)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    # Descriptions in italics under the ticks, angled to prevent overlap
    for i, desc in enumerate(PROGRESSION_DESCRIPTIONS):
        ax.text(i, -0.06, desc, ha="right", va="top", rotation=25,
                fontsize=9, style="italic", color="#333",
                transform=ax.get_xaxis_transform())

    ax.set_ylabel("mIoU on TotalSegmentator test set", fontsize=12)
    ax.set_ylim(0, 0.82)  # headroom for annotations
    ax.set_title("OrganCAM methodology progression (baseline → final)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

    # Annotations placed at fixed heights above the bars — no overlap.
    ax.annotate(
        "no retrain\n+0.059 mIoU",
        xy=(len(labels) - 2, values[-2] + 0.006),
        xytext=(len(labels) - 3.4, 0.76),
        fontsize=9, ha="center", color="#c1272d",
        arrowprops=dict(arrowstyle="->", color="#c1272d", lw=1.2),
    )
    ax.annotate(
        "final model\n+0.020 mIoU",
        xy=(len(labels) - 1, values[-1] + 0.006),
        xytext=(len(labels) - 0.4, 0.76),
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

    # Delta labels above each bar, anchored at fixed height so they don't
    # float away from small-value bars.
    for i, (u, t) in enumerate(zip(uniform, tuned)):
        delta = t - u
        col   = "#0a7a0a" if delta > 0.05 else "#555555"
        y_bar = max(u, t) + 0.015
        ax.text(x[i], y_bar, f"{'+' if delta >= 0 else ''}{delta:.2f}",
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
