"""Caption ablation study for OrganCAM.

Runs 5 prompt formulation levels through the fine-tuned model at inference
time to measure how prompt phrasing affects per-class mIoU. Text embeddings
are the only variable — no retraining.

Levels:
    L1 class_name        just the organ name  ("liver")
    L2 generic_clinical  short clinical caption  ("Axial CT scan of the liver.")
    L3 position_only     one-sentence position description (Claude-generated from L4)
    L4 full_anatomical   current FEW_SHOT_TEMPLATES (baseline used for training)
    L5 adversarial       L4 with left/right and top/bottom flipped (misleading)

Patch embeddings are cached once (single model forward per image) and reused
across all 5 levels for speed. Prompts are cached to diagnostics/ so re-runs
skip the Claude Haiku calls for L3.

Usage:
    python scripts/caption_ablation.py --checkpoint checkpoints/totalseg_patch_v4/epoch_015_val3.7378.pt --root Totalsegmentator_dataset_v201 --limit 300
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from open_clip import get_tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.totalsegmentator import load_totalsegmentator
from src.labels.templates import CLASS_NAMES, FEW_SHOT_TEMPLATES

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
OUT_DIR    = Path("diagnostics")

LEVELS      = [1, 2, 3, 4, 5]
LEVEL_NAMES = {1: "class_name", 2: "generic_clinical", 3: "position_only",
               4: "full_anatomical", 5: "adversarial"}


def load_model(checkpoint, device):
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)
    model = model.to(device)
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


def adversarial_swap(text: str) -> str:
    """Swap left/right and vertical position terms in the text (case-preserving)."""
    pairs = [
        ("LEFT",  "RIGHT"),
        ("Left",  "Right"),
        ("left",  "right"),
        ("upper", "lower"),
        ("Upper", "Lower"),
        ("top",   "bottom"),
        ("Top",   "Bottom"),
    ]
    # Use a placeholder pass to prevent double-swaps
    placeholders = {}
    for i, (a, b) in enumerate(pairs):
        p = f"__P{i}A__"; q = f"__P{i}B__"
        placeholders[p] = b; placeholders[q] = a
        text = text.replace(a, p).replace(b, q)
    for k, v in placeholders.items():
        text = text.replace(k, v)
    return text


def _call_claude(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def get_prompts(level: int, cache_dir: Path) -> dict[int, str]:
    cache = cache_dir / f"prompts_L{level}_{LEVEL_NAMES[level]}.json"
    if cache.exists():
        with open(cache) as f:
            return {int(k): v for k, v in json.load(f).items()}

    if level == 1:
        prompts = {c: name for c, name in CLASS_NAMES.items()}
    elif level == 2:
        prompts = {c: f"Axial CT scan of the {name.replace('-', ' ')}."
                   for c, name in CLASS_NAMES.items()}
    elif level == 3:
        print(f"  Generating L3 prompts via Claude Haiku (11 calls)...")
        prompts = {}
        for c, name in CLASS_NAMES.items():
            reply = _call_claude(
                f"Rewrite the following anatomical description as ONE SENTENCE that "
                f"describes ONLY the organ's position in the 2D CT image "
                f"(e.g. left/right side, top/bottom half, image center). "
                f"Do NOT mention appearance, morphology, adjacent structures, or "
                f"imaging characteristics. Output the sentence only, no preamble.\n\n"
                f"Description: {FEW_SHOT_TEMPLATES[c]}"
            )
            prompts[c] = reply
            print(f"    L3 {name}: {reply}")
    elif level == 4:
        prompts = dict(FEW_SHOT_TEMPLATES)
    elif level == 5:
        prompts = {c: adversarial_swap(FEW_SHOT_TEMPLATES[c]) for c in CLASS_NAMES}
    else:
        raise ValueError(level)

    with open(cache, "w") as f:
        json.dump({str(k): v for k, v in prompts.items()}, f, indent=2)
    return prompts


def build_patch_cache(model, dataset, n, device):
    features_buf = []
    handles = []
    for block_idx in [8, 9, 10, 11]:
        def _fwd(module, inp, out, _idx=block_idx):
            features_buf.append(out.detach())
        handles.append(model.visual.trunk.blocks[block_idx].register_forward_hook(_fwd))

    cache = []
    for idx in range(n):
        image, label, gt_mask = dataset[idx]
        features_buf.clear()
        with torch.no_grad():
            model.encode_image(image.unsqueeze(0).to(device))
            fused        = torch.stack(features_buf, dim=0).mean(dim=0)[0]   # (197, 768)
            patch_tokens = fused[1:, :]                                       # (196, 768)
            patch_emb    = F.normalize(model.visual.head(patch_tokens), dim=-1)  # (196, 512)
        cache.append((int(label), patch_emb, np.asarray(gt_mask) > 0.5))
        if (idx + 1) % 50 == 0 or (idx + 1) == n:
            print(f"  cached {idx+1}/{n}", flush=True)

    for h in handles:
        h.remove()
    return cache


def evaluate_level(cache, text_features):
    """Return {class_idx: {iou, dice, thr, n}} + overall mIoU, mDice."""
    masks_per_class: dict[int, list] = defaultdict(list)
    for label, patch_emb, gt in cache:
        sims    = patch_emb @ text_features[label]           # (196,)
        sim_map = sims.reshape(14, 14).unsqueeze(0).unsqueeze(0)
        mask    = F.interpolate(sim_map, size=(224, 224), mode="bilinear",
                                align_corners=False).squeeze().cpu().numpy()
        masks_per_class[label].append((mask, gt))

    per_class = {}
    all_ious, all_dice = [], []
    for c in sorted(masks_per_class):
        pairs = masks_per_class[c]
        lo = float(min(m.min() for m, _ in pairs))
        hi = float(max(m.max() for m, _ in pairs))
        candidates = np.linspace(lo, hi, 100)

        best_thr, best_iou = lo, -1.0
        for thr in candidates:
            ious = []
            for m, gt in pairs:
                if not gt.any():
                    continue
                pred  = m >= thr
                inter = (pred & gt).sum()
                union = (pred | gt).sum()
                if union > 0:
                    ious.append(inter / union)
            miou = float(np.mean(ious)) if ious else 0.0
            if miou > best_iou:
                best_iou, best_thr = miou, float(thr)

        ious, dices = [], []
        for m, gt in pairs:
            if not gt.any():
                continue
            pred  = m >= best_thr
            inter = (pred & gt).sum()
            union = (pred | gt).sum()
            denom = pred.sum() + gt.sum()
            if union > 0:
                ious.append(inter / union)
            if denom > 0:
                dices.append(2 * inter / denom)

        per_class[c] = {
            "iou":  float(np.mean(ious))  if ious  else float("nan"),
            "dice": float(np.mean(dices)) if dices else float("nan"),
            "thr":  best_thr,
            "n":    len(ious),
        }
        all_ious.extend(ious)
        all_dice.extend(dices)

    miou  = float(np.mean(all_ious))  if all_ious  else float("nan")
    mdice = float(np.mean(all_dice)) if all_dice else float("nan")
    return per_class, miou, mdice


def plot_heatmap(all_results: dict, out_path: Path):
    """Heatmap: bilateral pairs kept adjacent, groups ordered by difficulty.

    Sorting rows purely by mean IoU splits the pairs apart -- lung-left landed
    first and lung-right sixth, which buries the single most important
    comparison in this figure: the two lungs get the SAME adversarial
    transformation at L5 and respond very differently (lung-left 0.83->0.36,
    lung-right 0.78->0.63). Rows are therefore grouped so each left/right pair
    is adjacent, and the groups themselves are ordered by their mean IoU on the
    non-adversarial levels, preserving the easy-to-hard gradient.
    """
    from scripts._figstyle import apply_paper_style
    apply_paper_style(font_size=16)

    levels_all = sorted(all_results.keys())
    non_adv    = [l for l in levels_all if l != 5]

    class_mean_easy = {
        c: float(np.mean([all_results[l]["per_class"][c]["iou"] for l in non_adv]))
        for c in CLASS_NAMES
    }

    # Group bilateral pairs by their shared stem ("lung-left"/"lung-right" -> "lung"),
    # order groups by group-mean IoU, and keep left before right within a pair.
    def stem(c):
        n = CLASS_NAMES[c]
        return n.rsplit("-", 1)[0] if n.endswith(("-left", "-right")) else n

    groups = {}
    for c in CLASS_NAMES:
        groups.setdefault(stem(c), []).append(c)
    for members in groups.values():
        members.sort(key=lambda c: CLASS_NAMES[c])          # -left before -right
    classes = [
        c
        for _, members in sorted(
            groups.items(),
            key=lambda kv: -float(np.mean([class_mean_easy[c] for c in kv[1]])),
        )
        for c in members
    ]

    matrix = np.zeros((len(classes), len(levels_all)))
    for j, l in enumerate(levels_all):
        for i, c in enumerate(classes):
            matrix[i, j] = all_results[l]["per_class"][c]["iou"]

    # Drawn at 0.65\textwidth (4.2in), the width it is placed at, so LaTeX does
    # not rescale and the sizes below are literal points on the page. It sits
    # beside the metrics table rather than under it.
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(levels_all)))
    # Level number only. The formulation names live in the Methods
    # correspondence table, which the caption points at, and spelling them out
    # here cost the width that set the label size: "generic_clinical" caps the
    # labels near 6pt, "L2" has no width pressure at all.
    ax.set_xticklabels([f"L{l}" for l in levels_all], fontsize=9.5)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([CLASS_NAMES[c] for c in classes], fontsize=8)
    # Tick marks removed, so the labels would otherwise sit flush against the
    # heatmap edge. pad buys the gap the absent ticks used to provide.
    ax.tick_params(axis="both", length=0)
    ax.tick_params(axis="x", pad=6)
    ax.tick_params(axis="y", pad=5)

    for i in range(len(classes)):
        for j in range(len(levels_all)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.5 else "black", fontsize=7.5)

    # No embedded title: the LaTeX \caption already states both things this one
    # said (bilateral pairs adjacent, groups ordered easy to hard), and at this
    # width a two-line title costs vertical space for nothing.

    # Back on the right, vertical. It went horizontal when the x-axis carried
    # the formulation names and width set the label size; with bare L1--L5
    # labels width is no longer the binding constraint, and standing the bar up
    # buys back the vertical space, which now matters more -- the figure sits
    # beside a narrow table and a shorter figure balances that column better.
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("IoU", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=False)
    parser.add_argument("--root", required=False)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--replot-only", action="store_true",
                        help="Skip inference — read diagnostics/caption_ablation.json "
                             "and only regenerate the heatmap")
    args = parser.parse_args()

    if args.replot_only:
        json_path = OUT_DIR / "caption_ablation.json"
        if not json_path.exists():
            raise FileNotFoundError(f"{json_path} not found — run without --replot-only first")
        with open(json_path) as f:
            raw = json.load(f)
        all_results = {
            int(k): {**v, "per_class": {int(c): d for c, d in v["per_class"].items()}}
            for k, v in raw.items()
        }
        plot_heatmap(all_results, OUT_DIR / "caption_ablation.png")
        print(f"Regenerated {OUT_DIR / 'caption_ablation.png'}")
        return

    if not args.checkpoint or not args.root:
        parser.error("--checkpoint and --root are required unless --replot-only is set")

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Warning: ANTHROPIC_API_KEY not set — L3 generation will fail unless cached.")

    OUT_DIR.mkdir(exist_ok=True)

    print(f"Device: {args.device}")
    print("Loading model...")
    model     = load_model(args.checkpoint, args.device)
    tokenizer = get_tokenizer(MODEL_NAME)

    print("Loading test split...")
    _, _, test_ds = load_totalsegmentator(args.root)
    n = min(len(test_ds), args.limit)

    print(f"\nCaching patch embeddings for {n} images...")
    cache = build_patch_cache(model, test_ds, n, args.device)

    all_results = {}
    for level in LEVELS:
        print(f"\n== Level {level}: {LEVEL_NAMES[level]} ==")
        prompts       = get_prompts(level, OUT_DIR)
        pos_templates = [prompts[i] for i in range(len(CLASS_NAMES))]

        with torch.no_grad():
            text_features = F.normalize(
                model.encode_text(tokenizer(pos_templates).to(args.device)), dim=-1
            )

        per_class, miou, mdice = evaluate_level(cache, text_features)
        all_results[level] = {
            "name":      LEVEL_NAMES[level],
            "miou":      miou,
            "dice":      mdice,
            "per_class": {c: per_class[c] for c in per_class},
        }
        print(f"  mIoU={miou:.4f}  Dice={mdice:.4f}")
        for c in sorted(per_class):
            print(f"    {CLASS_NAMES[c]:<15} IoU={per_class[c]['iou']:.4f}"
                  f"  Dice={per_class[c]['dice']:.4f}  thr={per_class[c]['thr']:+.4f}")

    print("\n\n== Summary ==")
    print(f"{'Level':<20} {'mIoU':>8}  {'Dice':>8}")
    for level in LEVELS:
        r = all_results[level]
        print(f"L{level} {r['name']:<17} {r['miou']:>8.4f}  {r['dice']:>8.4f}")

    out_json = OUT_DIR / "caption_ablation.json"
    with open(out_json, "w") as f:
        json.dump({str(k): {**v, "per_class": {str(c): d for c, d in v["per_class"].items()}}
                   for k, v in all_results.items()}, f, indent=2)
    print(f"\nSaved {out_json}")

    plot_heatmap(all_results, OUT_DIR / "caption_ablation.png")
    print(f"Saved {OUT_DIR / 'caption_ablation.png'}")


if __name__ == "__main__":
    main()
