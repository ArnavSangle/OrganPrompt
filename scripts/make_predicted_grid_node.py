"""Panel A loss-box node: the PREDICTED 14x14 patch-similarity grid.

Pairs with node5_gt_mask_pooled.png. Those two arrays are exactly what the
per-patch BCE compares, so putting them side by side in the loss box shows the
training objective instead of describing it.

Faithfulness matters here, so this replicates src/training/patch_loss.py
rather than reusing the inference explainer:

    block 11 only (NOT the 8-11 fusion BioMedCLIPPatchSimilarity uses -- that
    is the inference path and belongs in panel B)
    -> drop CLS -> model.visual.head (768->512) -> L2 normalize
    -> cosine against the liver text embedding

Rendered as RAW cosine (logit_scale = 1), matching node8b's convention, so the
two similarity nodes in the figure are on comparable footing.

The slice is re-found with the same scan as node_1_and_5, so this is the same
slice node1/node5 already show.

Usage:
    python scripts/make_predicted_grid_node.py --device cuda
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.totalsegmentator import load_totalsegmentator          # noqa: E402
from src.tumor.organ_volume import load_organcam                     # noqa: E402
from scripts.make_pipeline_node_images import OUT, LIVER_CLASS       # noqa: E402


def find_liver_slice(root: str):
    """Same scan as node_1_and_5, so we land on the same slice it rendered."""
    print("[pred] loading TotalSegmentator to find the liver slice ...", flush=True)
    train_ds, _, _ = load_totalsegmentator(root, top_k=5, include_masks=True)

    best = None
    for i in range(len(train_ds)):
        item = train_ds.items[i]
        if item["organ_class"] != LIVER_CLASS:
            continue
        _, label, gt = train_ds[i]
        cov = float(gt.sum())
        if best is None or cov > best[0]:
            best = (cov, i)
        if len(train_ds.items) > 4000 and i > 4000:
            break

    assert best is not None, "no liver slice found -- check ORGAN_FILES / class index"
    cov, ds_idx = best
    print(f"[pred] training item {ds_idx}, liver coverage {cov:.0f} px")
    return train_ds[ds_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/TotalSeg/Totalsegmentator_dataset_v201")
    ap.add_argument("--checkpoint",
                    default="checkpoints/totalseg_patch_v5/epoch_011_val3.4584.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    img, _, gt = find_liver_slice(args.root)
    model, text_features = load_organcam(args.checkpoint, args.device)

    # --- replicate PatchContrastiveLoss.__call__ on this one slice ----------
    grabbed = {}
    block = model.visual.trunk.blocks[11]

    def _fwd(module, inp, out):
        grabbed["tokens"] = out[:, 1:, :].detach()   # (1, 196, 768), CLS dropped

    handle = block.register_forward_hook(_fwd)
    try:
        with torch.no_grad():
            model.encode_image(img.unsqueeze(0).to(args.device))
    finally:
        handle.remove()

    assert "tokens" in grabbed, "block 11 hook never fired"
    with torch.no_grad():
        patch_emb = F.normalize(model.visual.head(grabbed["tokens"]), dim=-1)  # (1,196,512)
        liver_t = text_features[LIVER_CLASS]                                   # (512,)
        sims = torch.einsum("bpd,d->bp", patch_emb, liver_t)                   # (1,196)
    cells = sims.squeeze(0).reshape(14, 14).cpu().numpy()

    # Same nearest-neighbour blow-up node5 and node8b use.
    blocky = np.kron(cells, np.ones((16, 16)))

    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    # Full 15-line grid with an explicit padded extent, matching node5: the two
    # sit side by side across the loss box, so they have to be drawn the same way.
    ax.imshow(blocky, cmap="viridis", extent=[0, 224, 224, 0])
    for k in range(15):
        ax.axhline(k * 16, color="#888888", linewidth=0.3)
        ax.axvline(k * 16, color="#888888", linewidth=0.3)
    ax.set_xlim(-0.6, 224.6)
    ax.set_ylim(224.6, -0.6)
    ax.axis("off")
    out = OUT / "node6b_predicted_grid.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)

    # Agreement between the predicted grid and the target the BCE uses, for
    # the caption -- reported, not tuned.
    gt_t = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pooled = F.interpolate(gt_t, size=(14, 14), mode="area").squeeze().numpy()
    target = pooled > 0.5
    print(f"[pred] wrote {out}")
    print(f"[pred] raw cosine range: {cells.min():.4f} .. {cells.max():.4f}")
    print(f"[pred] target liver cells: {target.sum()} / 196")
    print(f"[pred] mean sim inside liver:  {cells[target].mean():.4f}")
    print(f"[pred] mean sim outside liver: {cells[~target].mean():.4f}")


if __name__ == "__main__":
    main()
