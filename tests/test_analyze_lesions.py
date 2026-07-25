import numpy as np

import scripts.analyze_lesions as al


def _lesions(spec):
    """spec: {case: [rank, ...]} -> per_lesion dict with k=2 expectations."""
    out = {}
    for case, ranks in spec.items():
        for i, r in enumerate(ranks):
            out[f"{case}:{i}"] = {"case": case, "rank": r,
                                  "expected": ["lung-left", "lung-right"],
                                  "level": "T2"}
    return out


def test_bootstrap_resamples_patients_not_lesions():
    """The correction that makes n=184 honest.

    Two patients, 50 lesions each, every lesion within a patient identical. There
    are really only two independent observations. Resampling LESIONS would give a
    tight interval around 5 and imply precision that does not exist; resampling
    PATIENTS must reproduce the true uncertainty -- the mean is 1, 5 or 9
    depending on which patients are drawn.
    """
    per_lesion = _lesions({"a": [1] * 50, "b": [9] * 50})
    lo, hi = al.clustered_bootstrap_ci(per_lesion, n_boot=2000, seed=0)
    assert lo <= 1.5, f"lower bound {lo} too tight -- resampling lesions?"
    assert hi >= 8.5, f"upper bound {hi} too tight -- resampling lesions?"


def test_bootstrap_narrows_when_patients_actually_differ():
    """Sanity check on the same machinery: many independent patients whose ranks
    agree should give a tight interval, or the estimator is just noisy."""
    per_lesion = _lesions({f"p{i}": [3] for i in range(60)})
    lo, hi = al.clustered_bootstrap_ci(per_lesion, n_boot=2000, seed=0)
    assert lo == 3.0 and hi == 3.0


def test_permutation_uses_each_lesions_own_expected_set_size():
    """Chance depends on how many organs a level expects.

    An L1 lesion expects 5 of 11 organs, so rank<=2 is the coin-flip outcome; a
    T2 lesion expects 2 and chance is 4.0. Scoring both against one flat baseline
    inflates the lumbar levels, which is exactly the error that made an earlier
    per-patient result look stronger than it was.
    """
    small = {"a:0": {"case": "a", "rank": 2, "expected": ["x", "y"], "level": "T2"}}
    big = {"a:0": {"case": "a", "rank": 2,
                   "expected": list("abcde"), "level": "L1"}}
    _, chance_small, _ = al.clustered_permutation_p(small, n_perm=2000, seed=0)
    _, chance_big, _ = al.clustered_permutation_p(big, n_perm=2000, seed=0)
    assert chance_small == al.chance_rank(2)
    assert chance_big == al.chance_rank(5)
    assert chance_small > chance_big


def test_permutation_is_significant_when_ranks_beat_chance():
    per_lesion = _lesions({f"p{i}": [1] for i in range(40)})
    obs, chance, p = al.clustered_permutation_p(per_lesion, n_perm=2000, seed=0)
    assert obs == 1.0
    assert chance == al.chance_rank(2)
    assert p < 0.01


def test_permutation_is_not_significant_for_chance_level_ranks():
    rng = np.random.RandomState(0)
    ranks = [int(rng.permutation(11)[:2].min()) + 1 for _ in range(60)]
    per_lesion = _lesions({f"p{i}": [r] for i, r in enumerate(ranks)})
    _, _, p = al.clustered_permutation_p(per_lesion, n_perm=2000, seed=0)
    assert p > 0.05


def test_per_level_breakdown_groups_by_vertebral_level():
    per_lesion = {
        "a:0": {"case": "a", "rank": 1, "expected": ["x", "y"], "level": "T2"},
        "b:0": {"case": "b", "rank": 3, "expected": ["x", "y"], "level": "T2"},
        "c:0": {"case": "c", "rank": 2, "expected": list("abcde"), "level": "L1"},
    }
    got = al.by_level(per_lesion)
    assert got["T2"]["n"] == 2 and got["T2"]["mean_rank"] == 2.0
    assert got["L1"]["n"] == 1 and got["L1"]["mean_rank"] == 2.0
