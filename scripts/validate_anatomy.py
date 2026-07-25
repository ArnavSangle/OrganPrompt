"""Anatomical validation: does the per-organ proximity ranking track anatomy?

The bone-metastasis index is only meaningful if the organs it flags as "close"
are the organs that are actually close. Patients with exactly ONE involved
vertebral level give a free ground-truth test: a T2-only lesion should rank the
lungs near the top, an L4-only lesion the kidneys, an L5 lesion the bladder.

For each single-level patient we build the organ volume on the FULL CT (no crop
-- cropping around the tumor can hide an organ's true global peak and fabricate a
nearby one), compute per-organ proximity, and record the best rank achieved by
any anatomically expected organ. Random ranking would give a mean rank of ~6 out
of 11; a working index should be far better.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.spine_mets_labels import load_lesion_labels
import scripts.compute_index as ci

# Vertebral level -> organs whose typical craniocaudal extent overlaps that level.
# Deliberately generous (an organ spans several levels) but never trivial: no
# level expects more than 4 of the 11 classes, and the sets are disjoint between
# upper thoracic and lumbar. Cervical levels expect nothing in our 11-organ set
# and are excluded from the test entirely.
_LUNGS = ["lung-left", "lung-right"]
_KIDNEYS = ["kidney-left", "kidney-right"]
EXPECTED = {
    "T1": _LUNGS, "T2": _LUNGS, "T3": _LUNGS,
    "T4": _LUNGS + ["heart"], "T5": _LUNGS + ["heart"], "T6": _LUNGS + ["heart"],
    "T7": _LUNGS + ["heart"], "T8": _LUNGS + ["heart"],
    "T9": ["heart", "liver", "spleen"] + _LUNGS,
    "T10": ["liver", "spleen"] + _LUNGS,
    "T11": ["liver", "spleen", "pancreas"],
    "T12": ["liver", "spleen", "pancreas"] + _KIDNEYS,
    "L1": ["liver", "spleen", "pancreas"] + _KIDNEYS,
    "L2": ["pancreas"] + _KIDNEYS,
    "L3": _KIDNEYS,
    "L4": _KIDNEYS + ["bladder"],
    "L5": ["bladder", "femur-left", "femur-right"],
    "S1": ["bladder", "femur-left", "femur-right"],
}


def single_level_cases(labels):
    """{case: level} for patients with exactly one involved, testable level."""
    return {c: next(iter(lv)) for c, lv in labels.items()
            if len(lv) == 1 and next(iter(lv)) in EXPECTED}


def rank_of_expected(per_organ, expected):
    """1-based rank of the best-placed expected organ, by descending proximity."""
    order = sorted(per_organ, key=lambda n: per_organ[n]["proximity"], reverse=True)
    for i, name in enumerate(order, 1):
        if name in expected:
            return i
    return None


def summarize(per_case):
    ranks = [v["rank"] for v in per_case.values() if v["rank"] is not None]
    n = len(per_case)
    return {
        "n": n,
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
    ap.add_argument("--out_dir", default="results/anatomy_validation")
    ap.add_argument("--max_cases", type=int, default=0, help="0 = all")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    labels = load_lesion_labels(str(Path(a.root) / "Spine-Mets-CT-SEG_Clinical.xlsx"))
    cases_dir = Path(a.root) / "spine_mets_ct_seg"
    on_disk = {c.name for c in cases_dir.iterdir() if c.is_dir()}
    todo = {c: lv for c, lv in single_level_cases(labels).items() if c in on_disk}
    todo = dict(sorted(todo.items()))
    if a.max_cases:
        todo = dict(list(todo.items())[:a.max_cases])
    print(f"[validate_anatomy] {len(todo)} single-level patients: "
          + ", ".join(f"{c}={lv}" for c, lv in todo.items()), flush=True)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_case = {}
    for i, (case, level) in enumerate(todo.items()):
        expected = set(EXPECTED[level])
        cached = out_dir / f"{case}.json"
        print(f"[validate_anatomy] ({i + 1}/{len(todo)}) {case} level={level} "
              f"expected={sorted(expected)}", flush=True)
        try:
            if a.force or not cached.exists():
                # limit_slices=-1 -> full volume, no crop
                ci.run(a.root, case, a.checkpoint, a.device, 0.5, 0.5, -1, str(out_dir))
            data = json.loads(cached.read_text())
        except Exception as e:
            print(f"[validate_anatomy] {case}: SKIP ({e})", flush=True)
            continue
        r = rank_of_expected(data["per_organ"], expected)
        order = sorted(data["per_organ"],
                       key=lambda n: data["per_organ"][n]["proximity"], reverse=True)
        per_case[case] = {"level": level, "expected": sorted(expected),
                          "rank": r, "ranking": order}
        print(f"[validate_anatomy] {case}: best expected organ at rank {r} "
              f"| top3 = {order[:3]}", flush=True)

    summary = {"per_case": per_case, **summarize(per_case)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[validate_anatomy] n={summary['n']} top1={summary['top1']:.2f} "
          f"top3={summary['top3']:.2f} mean_rank={summary['mean_rank']:.2f} "
          f"(chance mean_rank ~ 4-6 depending on expected-set size)", flush=True)
    print(f"[validate_anatomy] wrote {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
