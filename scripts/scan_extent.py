"""Does the scan contain a pelvis where the model claims a bladder or a femur?

Reads results/lesion_validation/summary.json, pulls out the impossible bladder
and femur detections, and measures the craniocaudal coverage of every patient
from DICOM headers. Header-only: no pixels, no model, no GPU.

Interpretation is fixed here, before the numbers land, so the answer cannot be
chosen after the fact:

  If the phantom cases systematically LACK pelvic coverage, a field-of-view
  constraint removes them and is justified by physics -- an organ outside the
  imaged volume cannot be present. This is not the min-size filter rejected
  twice before, which tried to infer absence from a mask's shape.

  If the phantom cases have pelvic coverage indistinguishable from the rest,
  the field-of-view story is dead and the remaining explanation is that a real
  bladder lost to a chest phantom on peak response -- an assignment problem the
  z-score calibration A/B is the right experiment for.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.scan_extent import scan_extent, level_rank

FAMILIES = ("bladder", "femur")


def _family(organ):
    """'femur-left' -> 'femur'; 'liver' -> 'liver'."""
    head, _, tail = organ.rpartition("-")
    return head if tail in ("left", "right") and head else organ


def phantom_lesions(per_lesion, families=FAMILIES, max_mm=50.0,
                    thoracic_only=False):
    """Impossible top-1 detections that are physically close to the lesion.

    Distance matters. Proximity is 1/(1+d), so a bladder winning from 400mm is
    a scoring artifact and tells us nothing; one winning from 12mm means a
    bladder-labelled mask is sitting next to a thoracic vertebra.
    """
    out = []
    for key, v in per_lesion.items():
        ranking = v.get("ranking") or []
        if not ranking:
            continue
        top = ranking[0]
        fam = _family(top)
        if fam not in families or top in set(v["expected"]):
            continue
        d = v.get("distances_mm", {}).get(top)
        if d is None or d >= max_mm:
            continue
        if thoracic_only and not str(v["level"]).startswith("T"):
            continue
        out.append({"key": key, "case": v["case"], "level": v["level"],
                    "organ": top, "family": fam, "distance_mm": d})
    return out


def _fmt(x, nd=0):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/lesion_validation/summary.json")
    ap.add_argument("--root", default="data/Spine-Mets-CT-SEG_v1_2024")
    ap.add_argument("--out", default="results/lesion_validation/scan_extent.json")
    ap.add_argument("--max_mm", type=float, default=50.0)
    a = ap.parse_args()

    per = json.loads(Path(a.summary).read_text())["per_lesion"]
    phantoms = phantom_lesions(per, max_mm=a.max_mm)
    thoracic = [p for p in phantoms if str(p["level"]).startswith("T")]
    bad_cases = {p["case"] for p in phantoms}
    all_cases = sorted({v["case"] for v in per.values()})

    print(f"[scan_extent] {len(phantoms)} bladder/femur phantoms "
          f"({len(thoracic)} thoracic) across {len(bad_cases)} of "
          f"{len(all_cases)} patients\n", flush=True)

    cases_dir = Path(a.root) / "spine_mets_ct_seg"
    extents = {}
    for i, case in enumerate(all_cases):
        try:
            extents[case] = scan_extent(str(cases_dir / case))
        except Exception as e:
            print(f"[scan_extent] ({i + 1}/{len(all_cases)}) {case}: SKIP ({e})",
                  flush=True)
            continue
        e = extents[case]
        print(f"[scan_extent] ({i + 1}/{len(all_cases)}) {case}: "
              f"{e['highest_level']}..{e['lowest_level']}  "
              f"len={_fmt(e['length_mm'])}mm  "
              f"below-last-vertebra={_fmt(e['caudal_margin_mm'])}mm"
              f"{'   <- has phantoms' if case in bad_cases else ''}", flush=True)

    print("\n[scan_extent] phantom detections, with their scan's coverage:")
    for p in sorted(phantoms, key=lambda x: (x["case"], level_rank(x["level"]) or 0)):
        e = extents.get(p["case"])
        if e is None:
            continue
        print(f"  {p['key']:>12}  {p['organ']:>12} at {p['distance_mm']:5.0f}mm   "
              f"scan reaches {e['lowest_level']}, "
              f"{_fmt(e['caudal_margin_mm'])}mm below it")

    def _group(cases, field):
        vals = [extents[c][field] for c in cases
                if c in extents and extents[c][field] is not None]
        return np.array(vals, dtype=float)

    good_cases = [c for c in all_cases if c not in bad_cases]
    print("\n[scan_extent] coverage, phantom vs clean patients:")
    for field in ("caudal_margin_mm", "length_mm"):
        b, g = _group(sorted(bad_cases), field), _group(good_cases, field)
        if not len(b) or not len(g):
            continue
        print(f"  {field}:  phantom n={len(b)} median={np.median(b):7.1f}  "
              f"clean n={len(g)} median={np.median(g):7.1f}")

    lows = {}
    for c in all_cases:
        if c in extents:
            lows.setdefault(extents[c]["lowest_level"], []).append(c)
    print("\n[scan_extent] most caudal segmented vertebra, by patient count:")
    for lvl in sorted(lows, key=lambda x: level_rank(x) if level_rank(x) is not None else 99):
        n_bad = sum(c in bad_cases for c in lows[lvl])
        print(f"  {str(lvl):>4}: {len(lows[lvl]):3d} patients  ({n_bad} with phantoms)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"phantoms": phantoms, "extents": extents}, indent=2))
    print(f"\n[scan_extent] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
