"""Assess the per-lesion validation, with patient-level clustering.

Per-lesion scoring raises the evaluation set from 20 points to 184, but those 184
are not 184 independent trials: lesions within a patient share a scan, a set of
organ masks, and whatever that patient's anatomy happens to look like. Treating
them as independent would inflate significance by roughly the square root of the
average lesions-per-patient -- around 2x here -- which is a larger effect than
anything the method changes have produced.

So every uncertainty estimate below resamples PATIENTS, never lesions. The point
estimates use all 184 lesions; the confidence intervals are bounded by the 46
patients.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.analyze_validation import chance_rank, pair_splits


def _by_patient(per_lesion):
    """{case: [rank, ...]} keeping only ranked lesions."""
    out = {}
    for v in per_lesion.values():
        if v["rank"] is not None:
            out.setdefault(v["case"], []).append(v["rank"])
    return out


def clustered_bootstrap_ci(per_lesion, n_boot=5000, seed=0, alpha=0.05):
    """Percentile CI for mean rank, resampling PATIENTS with replacement.

    Resampling lesions would treat 50 lesions from one patient as 50 independent
    draws and report an interval far tighter than the data supports. The unit of
    independence is the patient, so that is the unit that gets resampled -- when a
    patient is drawn, all of its lesions come with it.
    """
    groups = list(_by_patient(per_lesion).values())
    if not groups:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    n = len(groups)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.randint(0, n, size=n)
        ranks = [r for i in pick for r in groups[i]]
        means[b] = np.mean(ranks)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def clustered_permutation_p(per_lesion, n_perm=20000, seed=0, n_organs=11):
    """(observed mean rank, chance mean rank, p), respecting per-lesion set size.

    The null draws a fresh random ranking for every lesion, so a patient with many
    lesions contributes many draws exactly as it does in the observed statistic.
    That keeps the comparison like-for-like; the clustering correction lives in
    the bootstrap interval, which is what a reader should quote alongside p.
    """
    ks, obs = [], []
    for v in per_lesion.values():
        if v["rank"] is None:
            continue
        ks.append(len(v["expected"]))
        obs.append(v["rank"])
    if not obs:
        return float("nan"), float("nan"), float("nan")
    observed = float(np.mean(obs))
    rng = np.random.RandomState(seed)
    hits = 0
    for _ in range(n_perm):
        tot = sum(rng.permutation(n_organs)[:k].min() + 1 for k in ks)
        if tot / len(ks) <= observed:
            hits += 1
    return observed, float(np.mean([chance_rank(k) for k in ks])), (hits + 1) / (n_perm + 1)


def by_level(per_lesion):
    """{level: {n, mean_rank, chance}} -- which vertebral levels actually work."""
    out = {}
    for v in per_lesion.values():
        if v["rank"] is None:
            continue
        out.setdefault(v["level"], {"ranks": [], "k": len(v["expected"])})
        out[v["level"]]["ranks"].append(v["rank"])
    return {lvl: {"n": len(d["ranks"]),
                  "mean_rank": float(np.mean(d["ranks"])),
                  "chance": chance_rank(d["k"])}
            for lvl, d in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/lesion_validation/summary.json")
    a = ap.parse_args()
    s = json.loads(Path(a.summary).read_text())
    per = s["per_lesion"]
    n_pat = len({v["case"] for v in per.values()})
    print(f"[analyze] {len(per)} lesions across {n_pat} patients "
          f"({len(per) / n_pat:.1f} per patient)\n")

    obs, chance, p = clustered_permutation_p(per)
    lo, hi = clustered_bootstrap_ci(per)
    print(f"[analyze] observed mean rank = {obs:.2f}  "
          f"95% CI [{lo:.2f}, {hi:.2f}]  (patient-clustered)")
    print(f"[analyze] chance mean rank   = {chance:.2f}")
    print(f"[analyze] permutation p      = {p:.4f}")
    print(f"[analyze] chance inside CI?  = "
          f"{'YES -- not distinguishable from chance' if lo <= chance <= hi else 'no'}\n")

    print("[analyze] by vertebral level:")
    lv = by_level(per)
    for level in sorted(lv, key=lambda x: (x[0], int(x[1:]))):
        d = lv[level]
        flag = "" if d["mean_rank"] < d["chance"] else "   <- at/below chance"
        print(f"  {level:>3}: n={d['n']:3d}  mean rank={d['mean_rank']:.2f}  "
              f"chance={d['chance']:.2f}{flag}")
    print()

    print("[analyze] anatomically impossible top-1 results:")
    bad = 0
    for c, v in per.items():
        if not v["ranking"]:
            continue
        top = v["ranking"][0]
        d = v["distances_mm"][top]
        if top not in v["expected"] and d < 50:
            print(f"  {c}: {top} at {d:.0f}mm -- not expected at {v['level']}")
            bad += 1
    print(f"  {bad}/{len(per)} lesions top-ranked an unexpected organ at <50mm\n")

    ps = pair_splits(per)
    print("[analyze] bilateral pair integrity:")
    print(f"  split = {ps['split']}/{ps['n_pairs']} "
          f"({ps['split'] / ps['n_pairs']:.0%})   <- must be 0")
    for pair, n in ps["by_pair"].items():
        if n:
            print(f"    {pair[0]}/{pair[1]}: {n} split")


if __name__ == "__main__":
    main()
