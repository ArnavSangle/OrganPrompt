"""Qualitative figure: one clean detection next to one phantom, side by side.

Unlike visualize_index.py (which picks the whole-patient peak tumor slice),
this targets a SPECIFIC (case, level) lesion and renders the actual
CompetitiveAssigner FINALIZED mask for the top-ranked organ at that lesion --
the same mask that produced the rank in results/lesion_validation/summary.json,
not a raw similarity heatmap. That makes the figure faithful to what actually
drove the result, including the connected-component/floor logic.

Cases were chosen by reading results/lesion_validation/summary.json directly
(see scratchpad query) for a clean, illustrative pair:
  clean:   10355:T3 -- lung-left ranked #1 at 19mm (expected: lung-left/right)
  phantom: 13683:T4 -- femur-right ranked #1 at 8mm (expected: heart/lung-left/lung-right)
"""
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

CASES = [
    {"case": "10355", "level": "T3", "top_organ": "lung-left",
     "expected": "lung-left / lung-right", "label": "correct detection",
     "distance_mm": 19.0},
    {"case": "13683", "level": "T4", "top_organ": "femur-right",
     "expected": "heart / lung-left / lung-right", "label": "phantom detection",
     "distance_mm": 8.0},
]

NAME_TO_CLASS = {v: k for k, v in CLASS_NAMES.items()}


def render_case(info, model, tf, device):
    """Raw per-organ similarity at the lesion's own slice -- not the finalized
    (whole-volume-dependent) mask. CompetitiveAssigner's peak/absence tracking
    is a whole-volume computation; restricting it to a small window around one
    lesion silently changes what counts as "peak" and "absent" and produces an
    empty mask that doesn't match the real per-lesion result. Raw similarity
    for a single slice has no such dependency -- matches what
    visualize_index.py already does, and is what actually drives the
    overlap/proximity scoring before the connected-component step.
    """
    labels = load_lesion_labels(str(Path(ROOT) / "Spine-Mets-CT-SEG_Clinical.xlsx"))
    p = load_patient(ROOT, info["case"], labels)
    coords = p.lesion_coords[info["level"]]
    z_peak = int(np.bincount(coords[:, 0]).argmax())

    print(f"[qualitative] {info['case']}:{info['level']} peak slice {z_peak} ...",
          flush=True)
    organ_class = NAME_TO_CLASS[info["top_organ"]]
    vol = build_organ_volume(p.ct[z_peak:z_peak + 1], model, tf, device,
                             organs=[organ_class])  # (1, 1, H, W)
    sim_slice = vol[0, 0]

    ct_disp = np.clip(p.ct[z_peak], -150, 250)
    ct_disp = (ct_disp + 150) / 400

    lesion_mask_slice = np.zeros(p.ct.shape[1:], dtype=bool)
    lesion_here = coords[coords[:, 0] == z_peak]
    lesion_mask_slice[lesion_here[:, 1], lesion_here[:, 2]] = True

    return ct_disp, lesion_mask_slice, sim_slice


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, tf = load_organcam(ci.DEFAULT_CKPT, "cuda")

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.6))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
    })

    for row, info in enumerate(CASES):
        ct_disp, lesion_mask, sim_slice = render_case(info, model, tf, "cuda")

        ax0, ax1 = axes[row]
        ax0.imshow(ct_disp, cmap="gray")
        ax0.imshow(np.ma.masked_where(~lesion_mask, lesion_mask), cmap="autumn", alpha=0.8)
        ax0.set_title(f"{info['case']}:{info['level']} — lesion", fontsize=9)
        ax0.axis("off")

        ax1.imshow(ct_disp, cmap="gray")
        # Raw similarity is routinely negative everywhere on a given slice (e.g.
        # femur-right here: -0.21 to -0.03, never positive) -- these organs win
        # RELATIVE to other organs' response at the same voxel, not in absolute
        # terms. An absolute-zero cutoff hides the real peak entirely. Highlight
        # the top 2% of THIS slice's own distribution instead, matching the
        # percentile-based territory selection compute_index.py itself uses.
        thresh = np.percentile(sim_slice, 98)
        sim_masked = np.ma.masked_where(sim_slice < thresh, sim_slice)
        im = ax1.imshow(sim_masked, cmap="viridis", alpha=0.65,
                        vmin=thresh, vmax=float(sim_slice.max()))
        color = "#0072B2" if info["label"] == "correct detection" else "#D55E00"
        ax1.set_title(f"top-ranked: {info['top_organ']} ({info['distance_mm']:.0f}mm)\n"
                      f"expected: {info['expected']}", fontsize=8, color=color)
        ax1.axis("off")

        axes[row, 0].text(-0.05, 0.5, info["label"].upper(), transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="center", fontsize=9, fontweight="bold",
                          color=color)

    fig.tight_layout()
    fig.savefig(OUT / "fig3_qualitative.svg", bbox_inches="tight", transparent=True)
    fig.savefig(OUT / "fig3_qualitative.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[qualitative] wrote {OUT / 'fig3_qualitative.png'}")


if __name__ == "__main__":
    main()
