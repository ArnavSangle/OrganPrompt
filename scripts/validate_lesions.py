"""Per-lesion anatomical validation: one evaluation point per labelled vertebra.

scripts/validate_anatomy.py scores one point per patient and can only use the 20
patients with exactly one involved level, because a patient with lesions at T2
and L4 has no single expected organ set. That ceiling is the binding constraint
on the whole Phase 1 result: with n=20, four assignment configurations have
already been scored on the same points, and removing the calibration leak moved p
from 0.0195 to 0.0529. There is no statistical room left to test anything else.

Scoring per LESION removes the ceiling. Each labelled vertebra carries its own
expectation -- a T2 lesion should rank the lungs, an L4 lesion the kidneys -- so
multi-level patients become usable and the evaluation set grows from 20 points to
184 across 46 patients.

The organ masks do not depend on which lesion is being scored, so each patient
costs exactly one inference pass regardless of how many lesions it carries.

Lesions within a patient share a scan and one set of organ masks, so they are NOT
independent observations. n=184 must never be quoted as 184 independent trials;
scripts/analyze_lesions.py resamples patients, not lesions.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.spine_mets_labels import load_lesion_labels
from src.data.spine_mets import load_patient, stride_coords
from src.tumor.organ_volume import load_organcam, iter_organ_similarity
from src.tumor.index import score_organ_against_lesions
from src.tumor.assign import CompetitiveAssigner
from src.labels.templates import CLASS_NAMES
from scripts.validate_anatomy import EXPECTED, detected, rank_of_expected, distances_mm
import scripts.compute_index as ci


def testable_lesions(labels, on_disk):
    """{case: [levels]} for every labelled level with an expected organ set.

    Unlike single_level_cases this keeps multi-level patients, which is the whole
    point -- they hold 164 of the 184 usable lesions.
    """
    out = {}
    for case, lv in labels.items():
        if case not in on_disk:
            continue
        levels = sorted(l for l in lv if l in EXPECTED)
        if levels:
            out[case] = levels
    return out


def score_patient(ct, lesion_coords, spacing, model, text_features, device,
                  floor=0.6, alpha=0.5, beta=0.5):
    """{level: {organ: record}} for one patient, from a single inference pass.

    Organ masks are streamed one at a time and each is scored against EVERY
    lesion before being discarded, so peak memory is one organ mask no matter how
    many lesions the patient carries.
    """
    organs = list(range(len(CLASS_NAMES)))
    assigner = CompetitiveAssigner(ct.shape)
    for oc, sim in iter_organ_similarity(ct, model, text_features, device, organs):
        assigner.update(oc, sim)
        del sim
    # Lesion voxels are vertebral bone by construction, so an organ predicted
    # there is a model artifact. Carve them out, exactly as compute_index does,
    # or a confused prediction at the lesion fakes overlap with itself.
    tumor = np.zeros(ct.shape, dtype=bool)
    for c in lesion_coords.values():
        tumor[c[:, 0], c[:, 1], c[:, 2]] = True

    per_lesion = {lvl: {} for lvl in lesion_coords}
    for oc, mask in assigner.finalize(floor=floor):
        mask &= ~tumor
        recs = score_organ_against_lesions(lesion_coords, mask, spacing, alpha, beta)
        for lvl, rec in recs.items():
            per_lesion[lvl][CLASS_NAMES[oc]] = rec
        del mask, recs
    return per_lesion


def summarize(per_lesion):
    """Headline numbers, reporting lesion and patient counts separately."""
    ranks = [v["rank"] for v in per_lesion.values() if v["rank"] is not None]
    n = len(per_lesion)
    return {
        "n_lesions": n,
        "n_patients": len({v["case"] for v in per_lesion.values()}),
        "n_ranked": len(ranks),
        "top1": sum(r <= 1 for r in ranks) / n if n else float("nan"),
        "top3": sum(r <= 3 for r in ranks) / n if n else float("nan"),
        "mean_rank": sum(ranks) / len(ranks) if ranks else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Spine-Mets-CT-SEG_v1_2024")
    ap.add_argument("--checkpoint", default=ci.DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="results/lesion_validation")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--floor", type=float, default=0.6)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--max_cases", type=int, default=0, help="0 = all")
    a = ap.parse_args()

    labels = load_lesion_labels(str(Path(a.root) / "Spine-Mets-CT-SEG_Clinical.xlsx"))
    cases_dir = Path(a.root) / "spine_mets_ct_seg"
    on_disk = {c.name for c in cases_dir.iterdir() if c.is_dir()}
    todo = dict(sorted(testable_lesions(labels, on_disk).items()))
    if a.max_cases:
        todo = dict(list(todo.items())[:a.max_cases])
    n_les = sum(len(v) for v in todo.values())
    print(f"[validate_lesions] {n_les} lesions across {len(todo)} patients", flush=True)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tf = load_organcam(a.checkpoint, a.device)

    per_lesion, skipped = {}, []
    for i, (case, levels) in enumerate(todo.items()):
        print(f"[validate_lesions] ({i + 1}/{len(todo)}) {case} levels={levels}",
              flush=True)
        try:
            p = load_patient(a.root, case, labels)
            ct = p.ct[::a.stride] if a.stride > 1 else p.ct
            spacing = (p.spacing[0] * a.stride, p.spacing[1], p.spacing[2])
            coords = {}
            for lvl in levels:
                c = stride_coords(p.lesion_coords.get(lvl, np.empty((0, 3), int)),
                                  a.stride)
                if len(c):
                    coords[lvl] = c
                else:
                    # a thin lesion can fall entirely on dropped slices
                    print(f"[validate_lesions]   {lvl}: SKIP (no voxels at "
                          f"stride {a.stride})", flush=True)
                    skipped.append(f"{case}:{lvl}")
            if not coords:
                continue
            scored = score_patient(ct, coords, spacing, model, tf, a.device,
                                   a.floor, a.alpha, a.beta)
            del p, ct
        except Exception as e:
            print(f"[validate_lesions] {case}: SKIP ({e})", flush=True)
            skipped.extend(f"{case}:{lvl}" for lvl in levels)
            continue

        for lvl, organs in scored.items():
            expected = set(EXPECTED[lvl])
            r = rank_of_expected(organs, expected)
            order = detected(organs)
            dmm = distances_mm(organs)
            per_lesion[f"{case}:{lvl}"] = {
                "case": case, "level": lvl, "expected": sorted(expected),
                "rank": r, "ranking": order,
                "distances_mm": {n: round(dmm[n], 1) for n in order},
                "absent": sorted(n for n, v in organs.items() if v["voxels"] == 0),
            }
            print(f"[validate_lesions]   {lvl}: rank {r} | "
                  + ", ".join(f"{n}={dmm[n]:.0f}mm" for n in order[:4]), flush=True)

    summary = {"per_lesion": per_lesion, "skipped": skipped,
               "stride": a.stride, "floor": a.floor, **summarize(per_lesion)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[validate_lesions] n_lesions={summary['n_lesions']} "
          f"n_patients={summary['n_patients']} top1={summary['top1']:.2f} "
          f"top3={summary['top3']:.2f} mean_rank={summary['mean_rank']:.2f}", flush=True)
    print(f"[validate_lesions] wrote {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
