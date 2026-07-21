"""Parse TrainingOutput.txt and generate training visualisation figures."""

import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

TRAINING_LOG = Path(__file__).parent.parent / "TrainingOutput.txt"
FIGURES_DIR  = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

EPOCH_SUMMARIES = [
    dict(epoch=1,  train=44.7540, val=3.5607, seconds=946),
    dict(epoch=2,  train=3.1561,  val=3.0882, seconds=944),
    dict(epoch=3,  train=3.0534,  val=3.1024, seconds=939),
    dict(epoch=4,  train=3.0188,  val=3.0721, seconds=938),
    dict(epoch=5,  train=2.9942,  val=3.0934, seconds=936),
    dict(epoch=6,  train=2.9732,  val=3.0552, seconds=936),
    dict(epoch=7,  train=2.9516,  val=3.0151, seconds=933),
    dict(epoch=8,  train=2.9485,  val=3.0213, seconds=935),
    dict(epoch=9,  train=2.9375,  val=3.0151, seconds=931),
    dict(epoch=10, train=2.9294,  val=3.0132, seconds=935),
]

plt.rcParams.update({"font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})


def parse_batch_losses():
    """Parse per-batch running-average loss per epoch from the log file."""
    epochs: dict[int, list[float]] = {}
    current_epoch = 0
    batch_re = re.compile(r"batch\s+(\d+)/\d+\s+\|\s+loss=([\d.]+)")
    with open(TRAINING_LOG, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = batch_re.search(line)
            if m:
                batch_num = int(m.group(1))
                loss      = float(m.group(2))
                if batch_num == 1:
                    current_epoch += 1
                    epochs[current_epoch] = []
                epochs.setdefault(current_epoch, []).append(loss)
    return epochs


def plot_epoch_curves():
    epochs = [d["epoch"] for d in EPOCH_SUMMARIES]
    train  = [d["train"] for d in EPOCH_SUMMARIES]
    val    = [d["val"]   for d in EPOCH_SUMMARIES]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(epochs, train, "o-", color="#2563EB", label="Train loss")
    ax.plot(epochs, val,   "s--", color="#DC2626", label="Val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Train vs Val Loss (all epochs)")
    ax.set_xticks(epochs)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.95))

    ax = axes[1]
    ax.plot(epochs[1:], train[1:], "o-", color="#2563EB", label="Train loss")
    ax.plot(epochs[1:], val[1:],   "s--", color="#DC2626", label="Val loss")
    best = min(EPOCH_SUMMARIES[1:], key=lambda d: d["val"])
    ax.scatter([best["epoch"]], [best["val"]], marker="*", s=260, color="gold",
               zorder=5, label=f"Best val {best['val']:.4f}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Train vs Val Loss (epochs 2–10, zoomed)")
    ax.set_xticks(epochs[1:])
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.95))

    fig.tight_layout()
    out = FIGURES_DIR / "epoch_loss_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_per_epoch_batch_curves(batch_losses):
    n      = len(batch_losses)
    cmap   = plt.get_cmap("tab10")
    colors = [cmap(i / n) for i in range(n)]
    sorted_epochs = sorted(batch_losses.items())

    # 10 individual subplots + 1 overlay = 11 total panels, 3 cols
    cols = 3
    rows = 4  # 11 panels in a 4×3 grid (last slot hidden)
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.5))
    axes = axes.flatten()

    for i, (epoch, losses) in enumerate(sorted_epochs):
        ax = axes[i]
        ax.plot(range(1, len(losses) + 1), losses,
                color=colors[i], linewidth=1.4)
        ax.set_title(f"Epoch {epoch}", fontsize=10, color=colors[i], fontweight="bold")
        ax.set_xlabel("Batch", fontsize=8)
        ax.set_ylabel("Running avg loss", fontsize=8)
        ax.tick_params(labelsize=7)

    # Panel 11: epochs 2–10 overlaid (epoch 1 excluded — warmup spike skews scale)
    ax_overlay = axes[n]
    for i, (epoch, losses) in enumerate(sorted_epochs):
        if epoch == 1:
            continue
        ax_overlay.plot(range(1, len(losses) + 1), losses,
                        color=colors[i], linewidth=1.2, alpha=0.85,
                        label=f"E{epoch}")
    ax_overlay.set_title("Epochs 2–10 overlaid", fontsize=10, fontweight="bold")
    ax_overlay.set_xlabel("Batch", fontsize=8)
    ax_overlay.set_ylabel("Running avg loss", fontsize=8)
    ax_overlay.tick_params(labelsize=7)
    ax_overlay.legend(frameon=False, fontsize=7, ncol=2,
                      loc="upper right", bbox_to_anchor=(1.0, 1.0))

    for j in range(n + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Per-batch running average loss — 10 epochs + overlay", y=1.01, fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / "per_epoch_batch_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_global_batch_loss(batch_losses):
    all_losses, epoch_boundaries = [], [0]
    for epoch in sorted(batch_losses):
        running = batch_losses[epoch]
        per_batch = []
        for k, avg in enumerate(running):
            per_batch.append(avg * (k + 1) - (running[k - 1] * k if k > 0 else 0))
        all_losses.extend(per_batch)
        epoch_boundaries.append(len(all_losses))

    fig, ax = plt.subplots(figsize=(14, 4.5))
    xs = np.arange(1, len(all_losses) + 1)
    ax.plot(xs, all_losses, linewidth=0.5, color="#2563EB", alpha=0.5)

    window = 50
    if len(all_losses) > window:
        smooth = np.convolve(all_losses, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window, len(all_losses) + 1), smooth,
                color="#DC2626", linewidth=1.8, label=f"{window}-batch moving avg")

    ymax = np.percentile(all_losses, 98)
    ax.set_ylim(bottom=0, top=ymax * 1.05)

    for b in epoch_boundaries[1:-1]:
        ax.axvline(b, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
    for i, b in enumerate(epoch_boundaries[:-1]):
        mid = (epoch_boundaries[i] + epoch_boundaries[i + 1]) // 2
        ax.text(mid, ymax * 0.97, f"E{i+1}", ha="center", fontsize=7, color="gray")

    ax.set_xlabel("Global batch number")
    ax.set_ylabel("Batch loss")
    ax.set_title("Loss across all batches (10 epochs)")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.95))
    fig.tight_layout()
    out = FIGURES_DIR / "global_batch_loss.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_epoch_timing():
    epochs  = [d["epoch"]   for d in EPOCH_SUMMARIES]
    minutes = [d["seconds"] / 60 for d in EPOCH_SUMMARIES]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(epochs, minutes, color="#2563EB", alpha=0.8, width=0.6)
    mean_excl1 = np.mean(minutes[1:])
    ax.axhline(mean_excl1, color="#DC2626", linestyle="--", linewidth=1.2,
               label=f"Mean (excl. epoch 1): {mean_excl1:.1f} min")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Time (minutes)")
    ax.set_title("Time per epoch")
    ax.set_xticks(epochs)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out = FIGURES_DIR / "epoch_timing.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    print("Parsing batch losses from TrainingOutput.txt...")
    batch_losses = parse_batch_losses()
    print(f"  Found {len(batch_losses)} epochs, "
          f"{sum(len(v) for v in batch_losses.values())} batch records")

    plot_epoch_curves()
    plot_per_epoch_batch_curves(batch_losses)
    plot_global_batch_loss(batch_losses)
    plot_epoch_timing()
    print("\nAll figures saved to figures/")
