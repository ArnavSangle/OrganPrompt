"""Exploratory subgroup table for the per-lesion validation (BSPC revision, R3 Q7).

Reviewer 3 asked for results by age, sex and tumor type. With 46 patients this
cannot support subgroup inference, so the table is descriptive only and carries
no p-values -- declared that way in docs/preregistration/2026-09-29-revision-rerun.md
before the rerun. Groups: sex; age split at the median of the evaluated
patients; primary cancer, pooling groups with fewer than `min_patients` patients
into "Other"; lesion type (lytic/blastic/mixed), which is per lesion.

Per group: patients, lesions, ranked lesions, mean rank over ranked lesions,
chance over the same lesions ((n+1)/(k+1) with each lesion's own k, exactly as
scripts/analyze_lesions.py), and top-1 over ALL lesions, as summary.json does.

    python scripts/subgroup_lesions.py --summary results/lesion_validation_v2/summary.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_validation import chance_rank

FACTORS = ("sex", "age", "primary", "lesion_type")


def load_clinical(xlsx_path):
    """{case: {age, sex, primary}} from the collection's clinical spreadsheet."""
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb["Spine-Mets-CT-SEG"]
    rows = ws.iter_rows(values_only=True)
    head = [str(h).strip() if h is not None else "" for h in next(rows)]
    ix = {name: head.index(name) for name in ("Case", "Age (Y)", "Sex", "Primary cancer")}
    out = {}
    for r in rows:
        if r[ix["Case"]] is None:
            continue
        out[str(r[ix["Case"]]).strip()] = {
            "age": float(r[ix["Age (Y)"]]),
            "sex": str(r[ix["Sex"]]).strip(),
            "primary": str(r[ix["Primary cancer"]]).strip()}
    return out


def _summarize(factor, group, lesions):
    ranked = [v for v in lesions if v["rank"] is not None]
    return {"factor": factor, "group": group,
            "n_patients": len({v["case"] for v in lesions}),
            "n_lesions": len(lesions), "n_ranked": len(ranked),
            "mean_rank": float(np.mean([v["rank"] for v in ranked])) if ranked else None,
            "chance": float(np.mean([chance_rank(len(v["expected"])) for v in ranked]))
                      if ranked else None,
            "top1": sum(1 for v in ranked if v["rank"] == 1) / len(lesions)}


def subgroup_table(per_lesion, clinical, lesion_types, min_patients=5):
    """List of group rows; every factor partitions all lesions exactly once."""
    lesions = list(per_lesion.values())
    for v in lesions:
        if v["case"] not in clinical:
            raise KeyError(f"no clinical record for case {v['case']}")
    cases = sorted({v["case"] for v in lesions})
    med = float(np.median([clinical[c]["age"] for c in cases]))
    n_primary = Counter(clinical[c]["primary"] for c in cases)

    def key(factor, v):
        c = clinical[v["case"]]
        if factor == "sex":
            return c["sex"]
        if factor == "age":
            return f"<= {med:g}" if c["age"] <= med else f"> {med:g}"
        if factor == "primary":
            return c["primary"] if n_primary[c["primary"]] >= min_patients else "Other"
        return lesion_types[v["case"]][v["level"]]

    rows = []
    for factor in FACTORS:
        groups = {}
        for v in lesions:
            groups.setdefault(key(factor, v), []).append(v)
        for g in sorted(groups, key=lambda g: (g == "Other", g)):
            rows.append(_summarize(factor, g, groups[g]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/lesion_validation_v2/summary.json")
    ap.add_argument("--root", default="data/Spine-Mets-CT-SEG_v1_2024")
    ap.add_argument("--min_patients", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from src.data.spine_mets_labels import load_lesion_labels
    xlsx = str(Path(a.root) / "Spine-Mets-CT-SEG_Clinical.xlsx")
    per = json.loads(Path(a.summary).read_text())["per_lesion"]
    rows = subgroup_table(per, load_clinical(xlsx), load_lesion_labels(xlsx), a.min_patients)
    out = Path(a.out or Path(a.summary).parent / "subgroups.json")
    out.write_text(json.dumps({"summary": a.summary, "exploratory": True, "rows": rows},
                              indent=2))
    print(f"{'factor':<12}{'group':<22}{'pts':>4}{'les':>5}{'rank':>7}{'chance':>8}{'top1':>6}")
    for r in rows:
        mr = f"{r['mean_rank']:.2f}" if r["mean_rank"] is not None else "--"
        ch = f"{r['chance']:.2f}" if r["chance"] is not None else "--"
        print(f"{r['factor']:<12}{r['group']:<22}{r['n_patients']:>4}{r['n_lesions']:>5}"
              f"{mr:>7}{ch:>8}{r['top1']:>6.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
