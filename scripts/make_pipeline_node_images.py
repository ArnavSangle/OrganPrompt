"""Generate the 5 real-data image crops for the hand-built pipeline schematic
(the other 6 nodes are hand-drawn icons in Google Drawings, not scripted).

Node 1  CT slice          -- TotalSegmentator liver slice, training pair
Node 5  GT organ mask      -- same slice, pooled to 14x14 exactly as
                              src/training/patch_loss.py does it (area
                              interpolation, threshold 0.5), rendered blocky
Node 8  Similarity map     -- Spine-Mets case 10355, T3, lung-left prompt
                              (the "clean detection" case from fig3)
Node 9  2D organ mask      -- same slice, thresholded to top 2%
Node 11 Metastasis index   -- case 10355 patient-level index (lung-left,
                              from results/bone_index/10355.json), peak
                              tumor slice, lesion + top-organ overlay

Nodes 8/9/11 reuse case 10355 throughout for continuity with fig3 and fig5,
which already use this case elsewhere in the paper.

Usage:
    python scripts/make_pipeline_node_images.py --device cuda
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts._figstyle import apply_paper_style
from src.data.totalsegmentator import load_totalsegmentator, HU_WINDOW, ORGAN_FILES
from src.data.spine_mets_labels import load_lesion_labels
from src.data.spine_mets import load_patient
from src.tumor.organ_volume import load_organcam, build_organ_volume
from src.labels.templates import CLASS_NAMES
from scripts.make_qualitative_figure import render_case, CASES as QUAL_CASES

OUT = Path("figures/paper/pipeline_nodes")
LIVER_CLASS = 6
SPINE_ROOT = "data/Spine-Mets-CT-SEG_v1_2024"


def node_1_and_5(root: str, device: str):
    """CT slice + its pooled GT mask, from one clean TotalSegmentator liver slice."""
    print("[node1/5] loading TotalSegmentator to find a clean liver slice ...")
    train_ds, _, _ = load_totalsegmentator(root, top_k=5, include_masks=True)

    best = None  # (coverage, ds_index)
    for i in range(len(train_ds)):
        item = train_ds.items[i]
        if item["organ_class"] != LIVER_CLASS:
            continue
        _, label, gt = train_ds[i]
        cov = float(gt.sum())
        if best is None or cov > best[0]:
            best = (cov, i)
        if len(train_ds.items) > 4000 and i > 4000:
            # Early volumes already give plenty of liver candidates; no need
            # to scan the full 35k-item training split for one example slice.
            break

    assert best is not None, "no liver slice found -- check ORGAN_FILES / class index"
    _, ds_idx = best
    img, label, gt = train_ds[ds_idx]
    print(f"[node1/5] using training item {ds_idx}, liver coverage {best[0]:.0f} px")

    # img is (3,224,224) HU-windowed and normalized to [0,1] by the dataset;
    # channel-repeated, so any channel is the real grayscale slice.
    ct_slice = img[0].numpy()

    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.imshow(ct_slice, cmap="gray")
    ax.axis("off")
    fig.savefig(OUT / "node1_ct_slice.png", dpi=200, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
    plt.close(fig)

    # Pool to 14x14 EXACTLY as src/training/patch_loss.py does: area
    # interpolation, then threshold at 0.5 -- not a visualization shortcut.
    gt_t = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pooled = F.interpolate(gt_t, size=(14, 14), mode="area").squeeze().numpy()
    pooled_binary = (pooled > 0.5).astype(np.float32)
    # Nearest-neighbor upscale for display only, so the 196 cells are visibly
    # blocky rather than a smoothed 14x14 thumbnail nobody can read.
    blocky = np.kron(pooled_binary, np.ones((16, 16)))

    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    # extent + explicit limits, because imshow's default limits are
    # [-0.5, 223.5]: the k=14 line at 224 would fall outside the axes and never
    # draw, and k=0 would be half-clipped, so the tile lost its right and bottom
    # border. The small pad keeps both outer lines at full width.
    ax.imshow(blocky, cmap="gray_r", vmin=0, vmax=1, extent=[0, 224, 224, 0])
    for k in range(15):
        ax.axhline(k * 16, color="#888888", linewidth=0.3)
        ax.axvline(k * 16, color="#888888", linewidth=0.3)
    ax.set_xlim(-0.6, 224.6)
    ax.set_ylim(224.6, -0.6)
    ax.axis("off")
    fig.savefig(OUT / "node5_gt_mask_pooled.png", dpi=200, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print("[node1/5] wrote node1_ct_slice.png, node5_gt_mask_pooled.png")


def node_8_and_9(model, tf, device: str):
    """Similarity map + thresholded mask, case 10355, T3, lung-left."""
    info = QUAL_CASES[0]  # {"case": "10355", "level": "T3", "top_organ": "lung-left", ...}
    assert info["case"] == "10355"
    print(f"[node8/9] rendering {info['case']}:{info['level']} ({info['top_organ']}) ...")
    ct_disp, lesion_mask, sim_slice = render_case(info, model, tf, device)

    thresh = np.percentile(sim_slice, 98)
    sim_masked = np.ma.masked_where(sim_slice < thresh, sim_slice)

    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.imshow(ct_disp, cmap="gray")
    ax.imshow(sim_masked, cmap="viridis", alpha=0.75, vmin=thresh,
              vmax=float(sim_slice.max()))
    ax.axis("off")
    fig.savefig(OUT / "node8_similarity_map.png", dpi=200, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
    plt.close(fig)

    mask_binary = (sim_slice >= thresh)
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.imshow(ct_disp, cmap="gray")
    ax.imshow(np.ma.masked_where(~mask_binary, mask_binary), cmap="viridis",
              alpha=0.85)
    ax.axis("off")
    fig.savefig(OUT / "node9_organ_mask.png", dpi=200, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print("[node8/9] wrote node8_similarity_map.png, node9_organ_mask.png")


def node_11(model, tf, device: str):
    """Patient-level metastasis index overlay, case 10355."""
    case = "10355"
    index_path = Path("results/bone_index") / f"{case}.json"
    d = json.loads(index_path.read_text())
    top_organ, top_stats = max(d["per_organ"].items(), key=lambda kv: kv[1]["score"])
    index_value = d["index"]
    print(f"[node11] case {case}: top organ {top_organ}, index {index_value:.3f} "
          f"(matches d['index']: {abs(index_value - top_stats['score']) < 1e-9})")

    labels = load_lesion_labels(str(Path(SPINE_ROOT) / "Spine-Mets-CT-SEG_Clinical.xlsx"))
    p = load_patient(SPINE_ROOT, case, labels)
    z = int(np.argmax(p.tumor_mask.sum(axis=(1, 2))))  # peak tumor slice

    organ_class = {v: k for k, v in CLASS_NAMES.items()}[top_organ]
    vol = build_organ_volume(p.ct[z:z + 1], model, tf, device, organs=[organ_class])
    sim = vol[0, 0]
    thresh = np.percentile(sim, 98)
    masked = np.ma.masked_where(sim < thresh, sim)

    ct_disp = np.clip(p.ct[z], -150, 250)
    ct_disp = (ct_disp + 150) / 400
    tm = p.tumor_mask[z]

    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    ax.imshow(ct_disp, cmap="gray")
    ax.imshow(np.ma.masked_where(~tm, tm), cmap="autumn", alpha=0.85)
    ax.imshow(masked, cmap="viridis", alpha=0.6, vmin=thresh, vmax=float(sim.max()))
    ax.text(0.02, 0.02, f"index = {index_value:.3f}", transform=ax.transAxes,
            fontsize=9, color="white", va="bottom", ha="left",
            bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none"))
    ax.axis("off")
    fig.savefig(OUT / "node11_metastasis_index.png", dpi=200, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print("[node11] wrote node11_metastasis_index.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/TotalSeg/Totalsegmentator_dataset_v201")
    ap.add_argument("--checkpoint", default="checkpoints/totalseg_patch_v5/epoch_011_val3.4584.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    apply_paper_style()
    OUT.mkdir(parents=True, exist_ok=True)

    node_1_and_5(args.root, args.device)

    print("[model] loading checkpoint for Spine-Mets nodes ...")
    model, tf = load_organcam(args.checkpoint, args.device)
    node_8_and_9(model, tf, args.device)
    node_11(model, tf, args.device)

    print(f"\nAll 5 node images written to {OUT}/")


if __name__ == "__main__":
    main()
