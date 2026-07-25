"""Assess the anatomy-validation output: proper chance baseline, permutation test,
and whether per-class peak-similarity gating removes phantom organ components.

Pure post-hoc analysis of results/anatomy_validation/summary.json -- no GPU, no
model, no re-running anything.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.validate_anatomy import EXPECTED


def chance_rank(k, n_organs=11):
    """Expected best-rank of k expected organs among n, if ranking were random.

    Order statistics: E[min of k draws without replacement from 1..n] = (n+1)/(k+1).
    This matters a lot here -- an L1 lesion expects 5 of 11 organs, so rank<=2 is
    the coin-flip outcome, not a success. Scoring every level against one flat
    "chance ~ 4-6" number silently inflates the result.
    """
    return (n_organs + 1) / (k + 1)


def permutation_p(per_case, n_perm=20000, seed=0):
    """P(random rankings achieve mean rank <= observed), respecting per-case k."""
    rng = np.random.RandomState(seed)
    ks, obs = [], []
    for v in per_case.values():
        if v["rank"] is None:
            continue
        ks.append(len(v["expected"]))
        obs.append(v["rank"])
    observed = float(np.mean(obs))
    n_organs = 11
    hits = 0
    for _ in range(n_perm):
        tot = 0.0
        for k in ks:
            # random ranking -> rank of best expected organ
            tot += rng.permutation(n_organs)[:k].min() + 1
        if tot / len(ks) <= observed:
            hits += 1
    return observed, float(np.mean([chance_rank(k) for k in ks])), (hits + 1) / (n_perm + 1)


def zscore_peaks(per_case):
    """Per-organ z-score of peak similarity across patients.

    Peak similarity is dominated by class identity (lungs always ~0.18, femurs
    always ~0.06), so an absolute floor cannot separate present from absent. The
    presence signal, if any, lives in an organ's deviation from its OWN norm.
    """
    names = sorted(next(iter(per_case.values()))["peak_sim"])
    mat = np.array([[per_case[c]["peak_sim"][n] for n in names] for c in per_case])
    mu, sd = mat.mean(axis=0), mat.std(axis=0)
    sd[sd == 0] = 1.0
    z = (mat - mu) / sd
    return names, dict(zip(per_case, z)), dict(zip(names, mu)), dict(zip(names, sd))


def gated_rank(case_v, zrow, names, thr):
    """Rank of best expected organ after dropping organs with z < thr.

    A gated-out expected organ is a MISS, not a skip -- otherwise the gate could
    trivially improve the metric by deleting the hard cases.
    """
    keep = {n for n, zz in zip(names, zrow) if zz >= thr}
    order = [n for n in case_v["ranking"] if n in keep]
    for i, n in enumerate(order, 1):
        if n in set(case_v["expected"]):
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/anatomy_validation/summary.json")
    a = ap.parse_args()
    s = json.loads(Path(a.summary).read_text())
    per_case = s["per_case"]
    print(f"[analyze] {len(per_case)} patients\n")

    obs, chance, p = permutation_p(per_case)
    print(f"[analyze] observed mean rank = {obs:.2f}")
    print(f"[analyze] chance mean rank   = {chance:.2f}  (NOT 4-6; expected sets are large)")
    print(f"[analyze] permutation p      = {p:.4f}\n")

    # per-level breakdown, since L1 (k=5) dominates the patient list
    by_k = {}
    for c, v in per_case.items():
        by_k.setdefault(len(v["expected"]), []).append(v["rank"])
    print("[analyze] by expected-set size:")
    for k in sorted(by_k):
        r = [x for x in by_k[k] if x is not None]
        print(f"  k={k}: n={len(r):2d}  mean rank={np.mean(r):.2f}  "
              f"chance={chance_rank(k):.2f}")
    print()

    # implausibility audit: distances that anatomy forbids
    print("[analyze] anatomically impossible top-1 results:")
    bad = 0
    for c, v in per_case.items():
        top = v["ranking"][0]
        d = v["distances_mm"][top]
        if top not in v["expected"] and d < 50:
            print(f"  {c} ({v['level']}): {top} at {d:.0f}mm -- not expected at this level")
            bad += 1
    print(f"  {bad}/{len(per_case)} patients top-ranked an unexpected organ at <50mm\n")

    names, zmap, mu, sd = zscore_peaks(per_case)
    print("[analyze] peak similarity by organ (mean +- sd across patients):")
    for n in sorted(names, key=lambda x: -mu[x]):
        print(f"  {n:14s} {mu[n]:.4f} +- {sd[n]:.4f}")
    print()

    print("[analyze] per-class z-score gating sweep:")
    print(f"  {'thr':>6} {'mean rank':>10} {'top1':>6} {'gated-out':>10} {'lost':>6}")
    for thr in [-99, -1.5, -1.0, -0.5, 0.0, 0.5]:
        ranks, lost, dropped = [], 0, 0
        for c, v in per_case.items():
            dropped += int((zmap[c] < thr).sum())
            r = gated_rank(v, zmap[c], names, thr)
            if r is None:
                lost += 1
            else:
                ranks.append(r)
        mr = np.mean(ranks) if ranks else float("nan")
        t1 = sum(r == 1 for r in ranks) / len(per_case)
        print(f"  {thr:>6} {mr:>10.2f} {t1:>6.2f} {dropped / len(per_case):>10.1f} {lost:>6d}")
    print("\n  (gated-out = organs dropped per patient; lost = patients where an "
          "expected\n   organ was gated away, counted as a miss)")


if __name__ == "__main__":
    main()
