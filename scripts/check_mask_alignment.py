"""Audit that every Spine-Mets vertebra mask sits on bone (BSPC revision, R5.3).

The SEG files store their column direction opposite to the CT's, and the loader
once pasted SEG frames onto the CT grid by z alone, mirroring every vertebra
anterior-posterior into the airway/mediastinum. src/data/spine_mets.py now maps
frames through both headers' geometry. This script is the all-patient evidence
for that fix: mean HU inside each vertebra mask as loaded, which is bone (well
above 150 HU) when the mask is placed correctly and air/soft tissue when it is
mirrored.

Header + pixel reads only; no model. Writes results/mask_alignment_audit.json.

    python scripts/check_mask_alignment.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pydicom

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.spine_mets import find_series, load_ct_volume, load_vertebra_masks

BONE_HU = 150.0


def audit_case(case_dir):
    """{level: mean HU} plus the two orientation headers for one patient."""
    ct_paths, seg_path = find_series(case_dir)
    vol, _ = load_ct_volume(ct_paths)
    masks = load_vertebra_masks(seg_path, ct_paths)
    ct_iop = [float(x) for x in pydicom.dcmread(
        ct_paths[0], stop_before_pixels=True).ImageOrientationPatient]
    seg = pydicom.dcmread(seg_path, stop_before_pixels=True)
    seg_iop = [float(x) for x in seg.SharedFunctionalGroupsSequence[0]
               .PlaneOrientationSequence[0].ImageOrientationPatient]
    levels = {lvl: float(vol[m].mean()) for lvl, m in masks.items()}
    return {"ct_iop": ct_iop, "seg_iop": seg_iop, "mean_hu": levels}


def failures(audit, threshold=BONE_HU):
    """[(case, level, mean HU)] for every level below the bone threshold."""
    return [(case, lvl, hu) for case, rec in sorted(audit.items())
            if "mean_hu" in rec
            for lvl, hu in sorted(rec["mean_hu"].items()) if hu < threshold]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Spine-Mets-CT-SEG_v1_2024/spine_mets_ct_seg")
    ap.add_argument("--out", default="results/mask_alignment_audit.json")
    a = ap.parse_args()
    cases = sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d)))
    audit = {}
    for i, case in enumerate(cases, 1):
        try:
            audit[case] = audit_case(os.path.join(a.root, case))
            hus = list(audit[case]["mean_hu"].values())
            print(f"[{i}/{len(cases)}] {case}: {len(hus)} levels, "
                  f"min mean HU {min(hus):.0f}", flush=True)
        except Exception as e:                     # record, don't hide
            audit[case] = {"error": repr(e)}
            print(f"[{i}/{len(cases)}] {case}: ERROR {e!r}", flush=True)
    bad = failures(audit)
    all_hu = [hu for r in audit.values() for hu in r.get("mean_hu", {}).values()]
    summary = {
        "n_cases": len(cases),
        "n_errors": sum("error" in r for r in audit.values()),
        "n_levels": len(all_hu),
        "threshold_hu": BONE_HU,
        "n_below_threshold": len(bad),
        "below_threshold": [{"case": c, "level": l, "mean_hu": h} for c, l, h in bad],
        "median_level_hu": float(np.median(all_hu)) if all_hu else None,
        "per_case": audit,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2))
    print(f"\n{summary['n_levels']} levels in {len(cases)} cases, "
          f"{len(bad)} below {BONE_HU:.0f} HU, {summary['n_errors']} errors -> {a.out}")


if __name__ == "__main__":
    main()
