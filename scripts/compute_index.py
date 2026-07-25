"""Compute the bone-metastasis organ index for one Spine-Mets patient."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.spine_mets_labels import load_lesion_labels
from src.data.spine_mets import load_patient
from src.tumor.organ_volume import load_organcam, iter_organ_similarity
from src.tumor.index import select_peak_component, patient_index
from src.labels.templates import CLASS_NAMES

DEFAULT_CKPT = "checkpoints/totalseg_patch_v5/epoch_011_val3.4584.pt"


def run(root, case, checkpoint, device, alpha, beta, limit_slices, out_dir, percentile=99.9,
        stride=1):
    print(f"[compute_index] loading clinical labels from {root} ...", flush=True)
    labels = load_lesion_labels(str(Path(root) / "Spine-Mets-CT-SEG_Clinical.xlsx"))
    if case is None:
        cases_dir = Path(root) / "spine_mets_ct_seg"
        case = next(c.name for c in sorted(cases_dir.iterdir())
                    if c.is_dir() and labels.get(c.name))
    print(f"[compute_index] loading patient {case} (CT + SEG + lesion labels) ...", flush=True)
    p = load_patient(root, case, labels)
    # crop to tumor-bearing slice range for speed
    # limit_slices < 0 means "no crop". A tight crop is a real hazard here: it can
    # exclude an organ's true global-peak slice, forcing select_peak_component to
    # latch onto a spurious local peak that happens to sit near the tumor.
    if limit_slices is not None and limit_slices >= 0:
        zs = np.where(p.tumor_mask.any(axis=(1, 2)))[0]
        z0, z1 = max(0, zs.min() - limit_slices), min(p.ct.shape[0], zs.max() + limit_slices + 1)
    else:
        z0, z1 = 0, p.ct.shape[0]
    ct = p.ct[z0:z1]
    tumor = p.tumor_mask[z0:z1]
    spacing = p.spacing
    if stride > 1:
        # Sparse subsampling through the full available range, NOT a range crop: real
        # organs span many consecutive slices, so skipping some still very likely keeps
        # a slice near an organ's true peak. Only used to make outlier huge-crop
        # patients tractable; z-spacing must scale with the stride for correct 3D
        # distances downstream.
        ct = ct[::stride]
        tumor = tumor[::stride]
        spacing = (spacing[0] * stride, spacing[1], spacing[2])
    print(f"[compute_index] cropped to slices [{z0}:{z1}] ({z1 - z0} of {p.ct.shape[0]}), "
          f"stride={stride} -> {ct.shape[0]} slices used; "
          f"tumor voxels = {int(tumor.sum())}, spacing = {spacing}", flush=True)
    if not tumor.any():
        raise ValueError(f"stride={stride} skipped every tumor slice for case {case}")
    organs = list(range(len(CLASS_NAMES)))
    model, tf = load_organcam(checkpoint, device)
    # Stream one organ at a time: the stacked [11,D,H,W] float32 volume is several GB
    # on an uncropped scan, and only the reduced boolean mask is ever needed.
    #
    # Fixed absolute thresholds (tuned on TotalSegmentator's single-organ-per-image
    # task) are far too permissive when scanning every organ across a huge cropped
    # volume: low-level false-positive coverage lands adjacent to the tumor almost
    # everywhere by pure chance. The single global-max-similarity voxel per organ is
    # reliably near the true organ location (verified empirically on this dataset), so
    # binarize per organ at a strict percentile and keep only the connected component
    # containing that peak — discard every other disconnected blob as noise.
    #
    # Spine-Mets tumor voxels are vertebral bone by construction (tumor_mask is a
    # union of vertebra SEG masks) — bone can never legitimately BE organ tissue, so
    # any organ classification firing on a tumor voxel is a model artifact, not
    # signal. Exclude it so a confused prediction at the lesion can't fake overlap.
    organ_bools = {}
    for oc, sim in iter_organ_similarity(ct, model, tf, device, organs):
        organ_bools[oc] = select_peak_component(sim, percentile) & ~tumor
        del sim
    # Report component SIZE, not coverage fraction: coverage is pinned at (100-pct)/100
    # by construction, so it says nothing about whether a component is plausible.
    organ_sizes = {CLASS_NAMES[oc]: int(ob.sum()) for oc, ob in organ_bools.items()}
    print("[compute_index] peak-component size (voxels): "
          + ", ".join(f"{n}={v}" for n, v in organ_sizes.items()), flush=True)
    res = patient_index(tumor, organ_bools, spacing, alpha, beta)
    res.update(case=case, spacing=list(spacing), tumor_voxels=int(tumor.sum()),
               alpha=alpha, beta=beta, percentile=percentile, stride=stride,
               slice_range=[int(z0), int(z1)], organ_sizes=organ_sizes)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case}.json"
    out_path.write_text(json.dumps(res, indent=2))
    print(f"[compute_index] index = {res['index']:.4f}", flush=True)
    return str(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Spine-Mets-CT-SEG_v1_2024")
    ap.add_argument("--case", default=None)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--limit_slices", type=int, default=8)
    ap.add_argument("--out_dir", default="results/bone_index")
    ap.add_argument("--percentile", type=float, default=99.9,
                     help="per-organ binarization percentile (see select_peak_component)")
    ap.add_argument("--stride", type=int, default=1,
                     help="use every Nth axial slice (z-spacing scales to match)")
    a = ap.parse_args()
    out = run(a.root, a.case, a.checkpoint, a.device, a.alpha, a.beta,
              a.limit_slices, a.out_dir, a.percentile, a.stride)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
