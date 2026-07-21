"""Diagnostic: dump raw patch-text similarity distributions per class.

For each of N test images, computes the 196 raw patch similarities (before
min-max normalization) and buckets them into foreground / background using
the 14×14 downsampled GT mask — the same target grid the patch loss used
during training.

Outputs:
    diagnostics/similarity_stats.json    per-class fg/bg mean, std, gap, Otsu threshold
    diagnostics/raw_similarities.png     fg vs bg histograms per class (raw)
    diagnostics/adjusted_similarities.png fg vs bg histograms per class (after neg subtraction)

Usage:
    python scripts/analyze_similarities.py --checkpoint checkpoints/totalseg_patch_v4/epoch_015_val3.7378.pt --root Totalsegmentator_dataset_v201 --limit 200
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import open_clip
import matplotlib.pyplot as plt
from open_clip import get_tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.totalsegmentator import load_totalsegmentator
from src.labels.templates import CLASS_NAMES, FEW_SHOT_TEMPLATES
from src.training.freeze import UNFREEZE_PATTERNS

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
OUT_DIR    = Path("diagnostics")


def load_model(checkpoint, device):
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)
    model = model.to(device)
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = Path(__file__).parent.parent / ckpt_path
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded {ckpt_path.name} (epoch {ckpt['epoch']}, val {ckpt['val_loss']:.4f})")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def register_multi_layer_hooks(model):
    """Hook ViT blocks 8-11 — same fusion as BioMedCLIPPatchSimilarity."""
    features = []
    handles = []
    for block_idx in [8, 9, 10, 11]:
        def _fwd(module, inp, out, _idx=block_idx):
            features.append(out.detach())
        handles.append(model.visual.trunk.blocks[block_idx].register_forward_hook(_fwd))
    return features, handles


@torch.no_grad()
def compute_patch_sims(model, image, text_features, target, features_buf):
    """Return raw (196,) similarity vector for target class."""
    features_buf.clear()
    model.encode_image(image)                                # fills features_buf
    fused = torch.stack(features_buf, dim=0).mean(dim=0)[0]  # (197, 768)
    patch_tokens = fused[1:, :]                              # (196, 768)
    patch_emb = F.normalize(model.visual.head(patch_tokens), dim=-1)  # (196, 512)
    sims = patch_emb @ text_features[target]                 # (196,)
    return sims


def otsu_threshold(fg, bg):
    """Threshold that maximizes between-class variance over pooled values."""
    values = np.concatenate([fg, bg])
    if values.size == 0:
        return float("nan")
    candidates = np.linspace(values.min(), values.max(), 100)
    best_thr, best_var = candidates[0], -1.0
    for thr in candidates:
        w0 = (values <  thr).mean()
        w1 = (values >= thr).mean()
        if w0 == 0 or w1 == 0:
            continue
        m0 = values[values <  thr].mean()
        m1 = values[values >= thr].mean()
        between = w0 * w1 * (m0 - m1) ** 2
        if between > best_var:
            best_var, best_thr = between, thr
    return float(best_thr)


def plot_distributions(per_class, key, out_path, title):
    """Grid of histograms — foreground (blue) vs background (red) per class."""
    n_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()
    for c in range(n_classes):
        ax = axes[c]
        fg = np.asarray(per_class[c][f"fg_{key}"])
        bg = np.asarray(per_class[c][f"bg_{key}"])
        if fg.size == 0 or bg.size == 0:
            ax.set_title(f"{CLASS_NAMES[c]} (no samples)")
            continue
        lo = min(fg.min(), bg.min())
        hi = max(fg.max(), bg.max())
        bins = np.linspace(lo, hi, 50)
        ax.hist(bg, bins=bins, alpha=0.55, color="crimson", density=True, label="background")
        ax.hist(fg, bins=bins, alpha=0.55, color="steelblue", density=True, label="foreground")
        thr = otsu_threshold(fg, bg)
        ax.axvline(thr, color="black", linestyle="--", linewidth=1, label=f"Otsu={thr:.3f}")
        ax.set_title(f"{CLASS_NAMES[c]}  gap={fg.mean() - bg.mean():+.3f}")
        ax.legend(fontsize=7)
    for c in range(n_classes, len(axes)):
        axes[c].axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    print(f"Device: {args.device}")
    print("Loading model...")
    model = load_model(args.checkpoint, args.device)

    print("Encoding text templates...")
    tokenizer     = get_tokenizer(MODEL_NAME)
    pos_templates = [FEW_SHOT_TEMPLATES[i] for i in range(len(CLASS_NAMES))]
    with torch.no_grad():
        text_features = F.normalize(
            model.encode_text(tokenizer(pos_templates).to(args.device)), dim=-1
        )  # (11, 512)

    print("Loading test split...")
    _, _, test_ds = load_totalsegmentator(args.root)
    n = min(len(test_ds), args.limit)
    print(f"Sampling {n} images\n")

    features_buf, hooks = register_multi_layer_hooks(model)

    per_class = defaultdict(lambda: {"fg_raw": [], "bg_raw": [], "fg_adj": [], "bg_adj": []})

    for idx in range(n):
        image, label, gt_mask = test_ds[idx]
        image_tensor = image.unsqueeze(0).to(args.device)

        raw_sims = compute_patch_sims(model, image_tensor, text_features, label, features_buf)  # (196,)

        # Negative subtraction: class-conditional (other 10 organ prompts)
        neg_idx = [i for i in range(len(CLASS_NAMES)) if i != label]
        # Recompute patch_emb once to avoid a second forward pass
        fused = torch.stack(features_buf, dim=0).mean(dim=0)[0]           # (197, 768)
        patch_emb = F.normalize(model.visual.head(fused[1:, :]), dim=-1)  # (196, 512)
        neg_sim   = (patch_emb @ text_features[neg_idx].T).max(dim=-1).values  # (196,)
        adj_sims  = (raw_sims - neg_sim).cpu().numpy()
        raw_sims  = raw_sims.cpu().numpy()

        # Downsample GT (224×224 → 14×14) matching patch loss target
        gt_t = torch.as_tensor(gt_mask).float().unsqueeze(0).unsqueeze(0)
        patch_labels = F.interpolate(gt_t, size=(14, 14), mode="area").squeeze().numpy().reshape(-1)
        fg_mask = patch_labels > 0.5

        if fg_mask.any() and (~fg_mask).any():
            per_class[label]["fg_raw"].extend(raw_sims[fg_mask].tolist())
            per_class[label]["bg_raw"].extend(raw_sims[~fg_mask].tolist())
            per_class[label]["fg_adj"].extend(adj_sims[fg_mask].tolist())
            per_class[label]["bg_adj"].extend(adj_sims[~fg_mask].tolist())

        if (idx + 1) % 50 == 0 or (idx + 1) == n:
            print(f"  {idx+1}/{n}", flush=True)

    for h in hooks:
        h.remove()

    # Summary table + JSON
    print(f"\n{'Class':<15} {'N_fg':>6} {'N_bg':>6}  {'fg_mean':>8} {'bg_mean':>8} {'gap':>6}  {'thr_raw':>7}  {'thr_adj':>7}")
    print("-" * 82)
    summary = {}
    for c in range(len(CLASS_NAMES)):
        fg_raw = np.asarray(per_class[c]["fg_raw"])
        bg_raw = np.asarray(per_class[c]["bg_raw"])
        fg_adj = np.asarray(per_class[c]["fg_adj"])
        bg_adj = np.asarray(per_class[c]["bg_adj"])
        if fg_raw.size == 0 or bg_raw.size == 0:
            summary[CLASS_NAMES[c]] = {"n_fg": 0, "n_bg": 0}
            print(f"{CLASS_NAMES[c]:<15} {'-':>6} {'-':>6}  {'-':>8} {'-':>8} {'-':>6}  {'-':>7}  {'-':>7}")
            continue
        thr_raw = otsu_threshold(fg_raw, bg_raw)
        thr_adj = otsu_threshold(fg_adj, bg_adj)
        summary[CLASS_NAMES[c]] = {
            "n_fg":     int(fg_raw.size),
            "n_bg":     int(bg_raw.size),
            "raw":      {"fg_mean": float(fg_raw.mean()), "bg_mean": float(bg_raw.mean()),
                         "fg_std":  float(fg_raw.std()),  "bg_std":  float(bg_raw.std()),
                         "otsu":    thr_raw},
            "adjusted": {"fg_mean": float(fg_adj.mean()), "bg_mean": float(bg_adj.mean()),
                         "fg_std":  float(fg_adj.std()),  "bg_std":  float(bg_adj.std()),
                         "otsu":    thr_adj},
        }
        gap = fg_raw.mean() - bg_raw.mean()
        print(f"{CLASS_NAMES[c]:<15} {fg_raw.size:>6} {bg_raw.size:>6}"
              f"  {fg_raw.mean():>+8.4f} {bg_raw.mean():>+8.4f} {gap:>+6.3f}"
              f"  {thr_raw:>+7.4f}  {thr_adj:>+7.4f}")

    summary_path = OUT_DIR / "similarity_stats.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nStats: {summary_path}")

    plot_distributions(per_class, "raw", OUT_DIR / "raw_similarities.png",
                       "Raw patch-text similarity (foreground vs background)")
    plot_distributions(per_class, "adj", OUT_DIR / "adjusted_similarities.png",
                       "After class-conditional negative subtraction")


if __name__ == "__main__":
    main()
