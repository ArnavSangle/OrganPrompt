"""Qualitative figure: one clean detection next to one phantom, side by side.

Unlike visualize_index.py (which picks the whole-patient peak tumor slice),
this targets a SPECIFIC (case, level) lesion and renders the actual
CompetitiveAssigner FINALIZED mask for the top-ranked organ at that lesion --
the same mask that produced the rank in results/lesion_validation/summary.json,
not a raw similarity heatmap. That makes the figure faithful to what actually
drove the result, including the connected-component/floor logic.

The two cases are fixed: 10355:T3 and 13683:T4, the pair shown in the submitted
paper and discussed by Reviewer 5. Their outcomes (top-ranked organ, distance,
whether it is anatomically expected) are read from the summary passed with
--summary, never hard-coded -- the submitted values came from mirrored masks.

    python scripts/make_qualitative_figure.py --summary results/lesion_validation_v2/summary.json
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.spine_mets_labels import load_lesion_labels
from src.data.spine_mets import load_patient
from src.tumor.organ_volume import load_organcam
from src.labels.templates import CLASS_NAMES
import scripts.compute_index as ci

ROOT = "data/Spine-Mets-CT-SEG_v1_2024"
OUT = Path("figures/paper")

CASE_KEYS = [("10355", "T3"), ("13683", "T4")]
UPLOAD = Path("paper/figures_for_upload")


def case_from_summary(summary, case, level):
    """Everything the panel titles state, read from the validation summary."""
    v = summary["per_lesion"][f"{case}:{level}"]
    top = v["ranking"][0]
    correct = top in v["expected"]
    return {"case": case, "level": level, "top_organ": top,
            "distance_mm": float(v["distances_mm"][top]),
            "expected": " / ".join(v["expected"]), "correct": correct,
            "label": "expected organ first" if correct else "unexpected organ first"}


NAME_TO_CLASS = {v: k for k, v in CLASS_NAMES.items()}


def pick_phantom_case(summary, organs=("femur-left", "femur-right")):
    """(case, level) of the first lesion, in case-ID then level order, whose
    top-ranked organ is one of `organs` and is not expected there. A fixed,
    neutral rule, stated before looking at the rendered panels."""
    for key in sorted(summary["per_lesion"]):
        v = summary["per_lesion"][key]
        if v["ranking"] and v["ranking"][0] in organs and v["ranking"][0] not in v["expected"]:
            return v["case"], v["level"]
    return None


def peak_slice(coords):
    """Axial index holding the most lesion voxels."""
    return int(np.bincount(np.asarray(coords)[:, 0]).argmax())


def nearest_pair(v_coords, o_coords, spacing):
    """(vertebra voxel, organ voxel, distance in mm) of the closest pair -- the
    quantity the ranking uses (src/tumor/index.py, KD-tree in mm space)."""
    from scipy.spatial import cKDTree
    sp = np.asarray(spacing, dtype=np.float64)
    v = np.asarray(v_coords)
    o = np.asarray(o_coords)
    d, j = cKDTree(o * sp).query(v * sp, k=1)
    i = int(np.argmin(d))
    return v[i], o[int(j[i])], float(d[i])


def render_case(info, model, tf, device, calib, stride=2, floor=0.6):
    """What the rank measured, drawn on one axial slice. The ASSIGNED mask of the
    top-ranked organ comes from the same whole-volume competitive assignment as
    scripts/validate_lesions.py (stride, cross-fitted calibration, floor,
    vertebra carve-out). The slice shown is the one holding the organ voxel
    nearest the involved vertebra; the vertebra is drawn as its outline projected
    over all its slices, and a segment joins the two closest points. Distance is
    3D, so the segment is its in-plane projection."""
    from scipy.ndimage import binary_erosion
    from src.data.spine_mets import stride_coords
    from src.tumor.assign import CompetitiveAssigner
    from src.tumor.organ_volume import iter_organ_similarity
    labels = load_lesion_labels(str(Path(ROOT) / "Spine-Mets-CT-SEG_Clinical.xlsx"))
    p = load_patient(ROOT, info["case"], labels)
    ct = p.ct[::stride]
    spacing = (p.spacing[0] * stride, p.spacing[1], p.spacing[2])
    coords = stride_coords(p.lesion_coords[info["level"]], stride)
    print(f"[qualitative] {info['case']}:{info['level']} assigning organs ...", flush=True)
    peaks, stats = calib(info["case"])
    assigner = CompetitiveAssigner(ct.shape, calibration="scale", peaks=peaks, stats=stats)
    for oc, sim in iter_organ_similarity(ct, model, tf, device, list(range(len(CLASS_NAMES)))):
        assigner.update(oc, sim)
        del sim
    tumor = np.zeros(ct.shape, dtype=bool)
    for c in p.lesion_coords.values():
        c = stride_coords(c, stride)
        if len(c):
            tumor[c[:, 0], c[:, 1], c[:, 2]] = True
    top = NAME_TO_CLASS[info["top_organ"]]
    organ = None
    for oc, mask in assigner.finalize(floor=floor):
        if oc == top:
            organ = mask & ~tumor
        del mask
    pv, po, d = nearest_pair(coords, np.argwhere(organ), spacing)
    z = int(po[0])
    vert_proj = np.zeros(ct.shape[1:], dtype=bool)
    vert_proj[coords[:, 1], coords[:, 2]] = True
    outline = vert_proj ^ binary_erosion(vert_proj, iterations=2)
    ct_disp = (np.clip(ct[z], -150, 250) + 150) / 400
    return {"ct": ct_disp, "organ": organ[z], "vertebra_outline": outline,
            "p_vertebra": (int(pv[1]), int(pv[2])), "p_organ": (int(po[1]), int(po[2])),
            "distance_mm": d, "slice_offset_mm": abs(int(po[0]) - int(pv[0])) * spacing[0]}


CACHE = Path("results/figure_cache")


def cached_render(info, model, tf, calib, refresh=False):
    """render_case output, saved per case so restyling the figure needs no model."""
    path = CACHE / f"fig7_{info['case']}_{info['level']}.npz"
    if path.exists() and not refresh:
        z = np.load(path)
        return {k: (tuple(int(x) for x in z[k]) if k.startswith("p_") else
                    float(z[k]) if z[k].ndim == 0 else z[k]) for k in z.files}
    r = render_case(info, model, tf, "cuda", calib)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **r)
    return r


def main():
    from matplotlib.colors import ListedColormap
    from src.tumor.crossfit import load_crossfit
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/lesion_validation_v2/summary.json")
    ap.add_argument("--crossfit", default="results/crossfit_folds.json")
    ap.add_argument("--organ_stats", default="results/organ_stats_all55.json")
    ap.add_argument("--refresh", action="store_true",
                    help="recompute the model output instead of using results/figure_cache")
    a = ap.parse_args()
    summary = json.loads(Path(a.summary).read_text())
    keys = list(CASE_KEYS)
    phantom = pick_phantom_case(summary)
    if phantom and phantom not in keys:
        keys.append(phantom)
    CASES = [case_from_summary(summary, c, l) for c, l in keys]
    calib = load_crossfit(a.crossfit, a.organ_stats)
    OUT.mkdir(parents=True, exist_ok=True)
    need_model = a.refresh or any(
        not (CACHE / f"fig7_{c['case']}_{c['level']}.npz").exists() for c in CASES)
    model, tf = load_organcam(ci.DEFAULT_CKPT, "cuda") if need_model else (None, None)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "svg.fonttype": "none",
    })
    # One panel per case at full printed width (6.5in): CT, involved vertebra
    # (red) and the assigned mask of the top-ranked organ (blue if expected at
    # that level, brick if not).
    fig, axes = plt.subplots(1, len(CASES), figsize=(6.5, 3.05))
    for ax, info in zip(np.atleast_1d(axes), CASES):
        r = cached_render(info, model, tf, calib, a.refresh)
        color = "#3B6EA5" if info["correct"] else "#B4543A"
        ax.imshow(r["ct"], cmap="gray")
        ax.imshow(np.ma.masked_where(~r["organ"], r["organ"]),
                  cmap=ListedColormap([color]), alpha=0.6)
        ax.imshow(np.ma.masked_where(~r["vertebra_outline"], r["vertebra_outline"]),
                  cmap=ListedColormap(["#E0301E"]), alpha=0.95)
        (vy, vx), (oy, ox) = r["p_vertebra"], r["p_organ"]
        ax.plot([vx, ox], [vy, oy], color="white", linewidth=0.9)
        ax.plot([vx, ox], [vy, oy], "o", color="white", markersize=2)
        ax.axis("off")
        verdict = "expected" if info["correct"] else "not expected"
        # the summary rounds to 0.1 mm; the figure recomputes the same distance
        print(f"[qualitative] {info['case']}:{info['level']} {info['top_organ']} "
              f"{r['distance_mm']:.1f} mm here vs {info['distance_mm']:.1f} mm in summary; "
              f"organ slice {r['slice_offset_mm']:.0f} mm from the vertebra's nearest slice")
        ax.set_title(f"{info['case']} {info['level']}\n"
                     f"ranked first: {info['top_organ']}\n"
                     f"{info['distance_mm']:.0f} mm, {verdict} at {info['level']}",
                     fontsize=9.5, color=color, linespacing=1.25)

    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "fig3_qualitative.svg", bbox_inches="tight", transparent=True)
    fig.savefig(OUT / "fig3_qualitative.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    UPLOAD.mkdir(parents=True, exist_ok=True)
    shutil.copy(OUT / "fig3_qualitative.png", UPLOAD / "detection_vs_phantom.png")
    print(f"[qualitative] cases: {', '.join(c['case'] + ':' + c['level'] for c in CASES)}")
    print(f"[qualitative] wrote {OUT / 'fig3_qualitative.png'} (+ copy in {UPLOAD})")


if __name__ == "__main__":
    main()
