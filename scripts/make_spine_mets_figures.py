"""Visualization figures for the bone-cancer (Spine-Mets) dataset.

Two figures, mirroring how the Phase 1 TotalSegmentator work was visualized
(figures/gradcam/gradcam_summary.png) but on bone-metastasis data:

  figures/paper/fig4_spine_organ_grid.{svg,png}
      11 organ columns x 3 rows (CT / patch-similarity / overlay) on a single
      Spine-Mets slice -- the direct analogue of the Grad-CAM summary figure,
      showing every organ map the index is computed from.

  figures/paper/fig5_spine_lesion_overlays.{svg,png}
      Multi-patient panel: lesion mask + top-ranked organ map, one column per
      patient, spanning the index-score range (low / mid / high) so the reader
      sees what different index values actually look like.

Usage:
    python scripts/make_spine_mets_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.spine_mets_labels import load_lesion_labels
from src.data.spine_mets import load_patient
from src.tumor.organ_volume import load_organcam, build_organ_volume
from src.labels.templates import CLASS_NAMES
import scripts.compute_index as ci

ROOT = "data/Spine-Mets-CT-SEG_v1_2024"
OUT = Path("figures/paper")

# Grid figure: a THORACIC lesion slice, not the whole-patient peak-tumor slice.
# 13055's peak tumor burden sits at the pelvis (slice 183), where heart, lungs,
# liver and spleen are simply not in frame -- their columns would show noise and
# imply the model hallucinates, when in fact the organs are absent. The T8 lesion
# slice has most of the 11 classes genuinely present, so each column is a fair
# test of whether the map lands on the right structure.
GRID_CASE = "13055"
GRID_LEVEL = "T8"

# Overlay figure: patients spanning the index-score distribution, chosen from
# results/bone_index/summary.json (low ~0.006, mid ~0.19, high 0.5).
OVERLAY_CASES = [
    ("14643", "low index (0.006)"),
    ("13627", "mid index (0.19)"),
    ("13653", "high index (0.50)"),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
})


def _ct_window(sl):
    """Soft-tissue window, matching visualize_index.py."""
    return (np.clip(sl, -150, 250) + 150) / 400


def _peak_tumor_slice(p):
    return int(np.argmax(p.tumor_mask.sum(axis=(1, 2))))


def fig4_organ_grid(model, tf, device):
    labels = load_lesion_labels(str(Path(ROOT) / "Spine-Mets-CT-SEG_Clinical.xlsx"))
    p = load_patient(ROOT, GRID_CASE, labels)
    coords = p.lesion_coords[GRID_LEVEL]
    z = int(np.bincount(coords[:, 0]).argmax())
    print(f"[spine_figs] grid: case {GRID_CASE} level {GRID_LEVEL}, slice {z}", flush=True)

    vol = build_organ_volume(p.ct[z:z + 1], model, tf, device)  # (11,1,H,W)
    ct = _ct_window(p.ct[z])

    n = len(CLASS_NAMES)
    fig, axes = plt.subplots(3, n, figsize=(1.35 * n, 4.4))
    for oc in range(n):
        sim = vol[oc, 0]
        # Same percentile highlight as fig3: raw similarity is routinely negative
        # everywhere, so organs win relatively, not absolutely -- an absolute
        # cutoff would render most columns blank.
        thresh = np.percentile(sim, 98)
        masked = np.ma.masked_where(sim < thresh, sim)

        axes[0, oc].imshow(ct, cmap="gray")
        axes[0, oc].set_title(CLASS_NAMES[oc], fontsize=7)
        axes[1, oc].imshow(sim, cmap="viridis")
        axes[2, oc].imshow(ct, cmap="gray")
        axes[2, oc].imshow(masked, cmap="viridis", alpha=0.65,
                           vmin=thresh, vmax=float(sim.max()))
        for r in range(3):
            axes[r, oc].axis("off")

    for r, lab in enumerate(["CT", "similarity", "overlay"]):
        axes[r, 0].text(-0.12, 0.5, lab, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=8,
                        fontweight="bold")

    fig.suptitle(f"OrganPrompt organ maps on bone-metastasis CT "
                 f"(Spine-Mets case {GRID_CASE}, {GRID_LEVEL} lesion slice {z})",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "fig4_spine_organ_grid.svg", bbox_inches="tight", transparent=True)
    fig.savefig(OUT / "fig4_spine_organ_grid.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[spine_figs] wrote {OUT / 'fig4_spine_organ_grid.png'}")


def fig5_lesion_overlays(model, tf, device):
    labels = load_lesion_labels(str(Path(ROOT) / "Spine-Mets-CT-SEG_Clinical.xlsx"))
    index = json.loads(Path("results/bone_index/summary.json").read_text())["per_case"]

    fig, axes = plt.subplots(2, len(OVERLAY_CASES),
                             figsize=(3.0 * len(OVERLAY_CASES), 6.2))
    for col, (case, label) in enumerate(OVERLAY_CASES):
        p = load_patient(ROOT, case, labels)
        z = _peak_tumor_slice(p)
        print(f"[spine_figs] overlay: case {case}, peak tumor slice {z}", flush=True)

        # Top-scoring organ at this lesion, by mean similarity inside the tumor.
        vol = build_organ_volume(p.ct[z:z + 1], model, tf, device)
        tm = p.tumor_mask[z]
        scores = [vol[o, 0][tm].mean() if tm.any() else vol[o, 0].mean()
                  for o in range(len(CLASS_NAMES))]
        top = int(np.argmax(scores))
        sim = vol[top, 0]
        thresh = np.percentile(sim, 98)
        masked = np.ma.masked_where(sim < thresh, sim)
        ct = _ct_window(p.ct[z])

        axes[0, col].imshow(ct, cmap="gray")
        axes[0, col].imshow(np.ma.masked_where(~tm, tm), cmap="autumn", alpha=0.85)
        axes[0, col].set_title(f"{case} — {label}\nlesion (slice {z})", fontsize=8)

        axes[1, col].imshow(ct, cmap="gray")
        axes[1, col].imshow(masked, cmap="viridis", alpha=0.65,
                            vmin=thresh, vmax=float(sim.max()))
        axes[1, col].set_title(f"nearest organ: {CLASS_NAMES[top]}\n"
                               f"index = {index[case]:.3f}", fontsize=8)
        for r in range(2):
            axes[r, col].axis("off")

    fig.suptitle("Bone-metastasis index across the score range", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig5_spine_lesion_overlays.svg", bbox_inches="tight", transparent=True)
    fig.savefig(OUT / "fig5_spine_lesion_overlays.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[spine_figs] wrote {OUT / 'fig5_spine_lesion_overlays.png'}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, tf = load_organcam(ci.DEFAULT_CKPT, "cuda")
    fig4_organ_grid(model, tf, "cuda")
    fig5_lesion_overlays(model, tf, "cuda")


if __name__ == "__main__":
    main()
