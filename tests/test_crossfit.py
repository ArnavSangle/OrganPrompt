"""Patient-level cross-fitting of the calibration constants (BSPC revision, R5.2)."""
import math

import pytest

from src.tumor.crossfit import make_folds, fold_peaks, fold_stats, fit_cases

EVAL = [f"c{i:02d}" for i in range(46)]
NOLES = [f"n{i}" for i in range(9)]


def test_folds_cover_every_eval_case_once():
    f = make_folds(EVAL, k=5, seed=0)
    assert set(f) == set(EVAL)
    assert set(f.values()) == set(range(5))


def test_folds_are_balanced():
    f = make_folds(EVAL, k=5, seed=0)
    sizes = [list(f.values()).count(i) for i in range(5)]
    assert max(sizes) - min(sizes) <= 1


def test_folds_are_deterministic_and_order_independent():
    assert make_folds(EVAL, 5, 0) == make_folds(list(reversed(EVAL)), 5, 0)


def test_fit_cases_exclude_own_fold_and_include_no_lesion_patients():
    f = make_folds(EVAL, 5, 0)
    for fold in range(5):
        fit = set(fit_cases(f, NOLES, fold))
        assert set(NOLES) <= fit
        assert not any(f[c] == fold for c in fit if c in f)
        assert fit | {c for c in EVAL if f[c] == fold} == set(EVAL) | set(NOLES)


def test_fold_peaks_never_sees_its_own_fold():
    f = make_folds(EVAL, 5, 0)
    per_case = {c: {"liver": 1.0} for c in EVAL + NOLES}
    for c in EVAL:
        if f[c] == 2:
            per_case[c]["liver"] = 999.0            # poison fold 2
    assert fold_peaks(per_case, f, NOLES, fold=2) == {"liver": 1.0}


def test_fold_peaks_is_mean_of_per_patient_peaks():
    f = {"a": 0, "b": 1, "c": 1}
    per_case = {"a": {"x": 9.0}, "b": {"x": 1.0}, "c": {"x": 3.0}, "n": {"x": 2.0}}
    assert fold_peaks(per_case, f, ["n"], fold=0) == {"x": 2.0}


def test_fold_stats_pools_moments_by_voxel_count():
    # fit = b + n: sum 14, sumsq 38, count 6 -- fold 0's own case a is excluded
    f = {"a": 0, "b": 1}
    moments = {"a": {"x": [100.0, 100.0, 1]},
               "b": {"x": [2.0, 2.0, 2]},
               "n": {"x": [12.0, 36.0, 4]}}
    (mean, sd), = fold_stats(moments, f, ["n"], fold=0).values()
    assert mean == pytest.approx(14.0 / 6)
    assert sd == pytest.approx(math.sqrt(38.0 / 6 - (14.0 / 6) ** 2))


def test_fold_peaks_rejects_a_fit_case_without_stats():
    with pytest.raises(KeyError):
        fold_peaks({"n": {"x": 1.0}}, {"a": 0, "b": 1}, ["n"], fold=0)


def test_load_crossfit_gives_each_case_its_own_fold_constants(tmp_path):
    import json
    from src.tumor.crossfit import load_crossfit
    folds = {"k": 2, "seed": 0, "folds": {"a": 0, "b": 1}, "always_fit": ["n"]}
    stats = {"per_case": {"a": {"x": 1.0}, "b": {"x": 3.0}, "n": {"x": 5.0}},
             "per_case_moments": {"a": {"x": [1.0, 1.0, 1]}, "b": {"x": [3.0, 9.0, 1]},
                                  "n": {"x": [5.0, 25.0, 1]}}}
    (tmp_path / "f.json").write_text(json.dumps(folds))
    (tmp_path / "s.json").write_text(json.dumps(stats))
    calib = load_crossfit(tmp_path / "f.json", tmp_path / "s.json")
    peaks_a, stats_a = calib("a")                 # fit on b + n
    assert peaks_a == {"x": 4.0}
    assert stats_a["x"][0] == pytest.approx(4.0)
    with pytest.raises(KeyError):
        calib("not-in-folds")


def test_folds_spec_puts_non_evaluated_patients_in_always_fit():
    from src.tumor.crossfit import folds_spec
    spec = folds_spec(eval_cases=["a", "b", "c"], all_cases=["a", "b", "c", "n1", "n2"],
                      k=2, seed=7)
    assert spec["always_fit"] == ["n1", "n2"]
    assert set(spec["folds"]) == {"a", "b", "c"}
    assert spec["k"] == 2 and spec["seed"] == 7
