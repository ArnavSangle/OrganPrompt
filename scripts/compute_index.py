"""Compute the bone-metastasis organ index for one Spine-Mets patient."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.spine_mets_labels import load_lesion_labels
from src.data.spine_mets import load_patient
from src.tumor.organ_volume import load_organcam, build_organ_volume
from src.tumor.index import binarize_organs, patient_index
from src.labels.templates import CLASS_NAMES

DEFAULT_CKPT = "checkpoints/totalseg_patch_v5/epoch_011_val3.4584.pt"


def _load_thresholds():
    p = Path("diagnostics/tuned_thresholds.json")
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    name_to_idx = {v: k for k, v in CLASS_NAMES.items()}
    out = {}
    for name, v in data.items():
        if name in name_to_idx and isinstance(v, (int, float)):
            out[name_to_idx[name]] = float(v)
    return out


def run(root, case, checkpoint, device, alpha, beta, limit_slices, out_dir):
    print(f"[compute_index] loading clinical labels from {root} ...", flush=True)
    labels = load_lesion_labels(str(Path(root) / "Spine-Mets-CT-SEG_Clinical.xlsx"))
    if case is None:
        cases_dir = Path(root) / "spine_mets_ct_seg"
        case = next(c.name for c in sorted(cases_dir.iterdir())
                    if c.is_dir() and labels.get(c.name))
    print(f"[compute_index] loading patient {case} (CT + SEG + lesion labels) ...", flush=True)
    p = load_patient(root, case, labels)
    # crop to tumor-bearing slice range for speed
    zs = np.where(p.tumor_mask.any(axis=(1, 2)))[0]
    z0, z1 = max(0, zs.min() - limit_slices), min(p.ct.shape[0], zs.max() + limit_slices + 1)
    ct = p.ct[z0:z1]
    tumor = p.tumor_mask[z0:z1]
    print(f"[compute_index] cropped to slices [{z0}:{z1}] ({z1 - z0} of {p.ct.shape[0]}); "
          f"tumor voxels = {int(tumor.sum())}", flush=True)
    organs = list(range(len(CLASS_NAMES)))
    model, tf = load_organcam(checkpoint, device)
    organ_vol = build_organ_volume(ct, model, tf, device, organs)
    thresholds = _load_thresholds()
    organ_bools = binarize_organs(organ_vol, thresholds, organs)
    # Spine-Mets tumor voxels are vertebral bone by construction (tumor_mask is a union
    # of vertebra SEG masks) — bone can never legitimately BE organ tissue, so any organ
    # classification firing on a tumor voxel is a model artifact, not signal. Exclude it
    # before scoring so a confused prediction at the lesion can't manufacture overlap.
    organ_bools = {oc: ob & ~tumor for oc, ob in organ_bools.items()}
    organ_coverage = {CLASS_NAMES[oc]: float(ob.mean()) for oc, ob in organ_bools.items()}
    print("[compute_index] organ mask coverage (fraction of cropped volume): "
          + ", ".join(f"{n}={v:.3f}" for n, v in organ_coverage.items()), flush=True)
    res = patient_index(tumor, organ_bools, p.spacing, alpha, beta)
    res.update(case=case, spacing=list(p.spacing), tumor_voxels=int(tumor.sum()),
               alpha=alpha, beta=beta, slice_range=[int(z0), int(z1)],
               organ_coverage=organ_coverage)
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
    a = ap.parse_args()
    out = run(a.root, a.case, a.checkpoint, a.device, a.alpha, a.beta,
              a.limit_slices, a.out_dir)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
