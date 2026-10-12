"""Patient-level cross-fitting of the calibration constants.

src/tumor/assign.py calibrates each organ's similarity by constants fitted on
Spine-Mets patients (TYPICAL_PEAK for "scale", ORGAN_MEAN/ORGAN_SD for
"zscore"). Those were fitted once, on the 35 patients outside the 20-patient
single-level evaluation -- but the per-lesion evaluation later grew to 46
patients, 26 of which were in that fitting set (Reviewer 5, BSPC revision).

Cross-fitting removes the overlap. The 46 evaluation patients are split into k
folds; each patient is scored with constants fitted on the OTHER folds plus the
9 no-lesion patients, which are never evaluated and so join every fold's fit.
No patient is ever calibrated on its own scan.

Inputs are the per-patient reductions scripts/fit_organ_stats.py records:
per_case {case: {organ: peak}} and per_case_moments {case: {organ: [sum,
sumsq, count]}}, so no model is run here.
"""
import json
import math
import random
from pathlib import Path

import numpy as np


def make_folds(eval_cases, k, seed):
    """{case: fold} -- sorted then seeded-shuffled, so input order is irrelevant."""
    cases = sorted(eval_cases)
    random.Random(seed).shuffle(cases)
    return {c: i % k for i, c in enumerate(cases)}


def fit_cases(folds, always_fit, fold):
    """Patients whose statistics calibrate `fold`: other folds + always_fit."""
    return sorted([c for c, f in folds.items() if f != fold] + list(always_fit))


def fold_peaks(per_case, folds, always_fit, fold):
    """{organ_name: mean per-patient peak} over fit_cases -- TYPICAL_PEAK's definition.

    Keyed by organ NAME, as CompetitiveAssigner(peaks=...) looks them up.
    Raises KeyError if a fitting patient has no recorded stats rather than
    quietly fitting on fewer patients than the fold says.
    """
    fit = fit_cases(folds, always_fit, fold)
    organs = sorted(per_case[fit[0]])
    return {o: float(np.mean([per_case[c][o] for c in fit])) for o in organs}


def fold_stats(per_case_moments, folds, always_fit, fold):
    """{organ_name: (mean, sd)} pooled over every voxel of fit_cases.

    Same pooling as OrganStatsAccumulator.result(): voxel-weighted, from raw
    moments, variance clamped at zero against cancellation.
    """
    fit = fit_cases(folds, always_fit, fold)
    out = {}
    for o in sorted(per_case_moments[fit[0]]):
        s = sum(per_case_moments[c][o][0] for c in fit)
        ss = sum(per_case_moments[c][o][1] for c in fit)
        n = sum(per_case_moments[c][o][2] for c in fit)
        mean = s / n
        out[o] = (float(mean), float(math.sqrt(max(ss / n - mean * mean, 0.0))))
    return out


def load_crossfit(folds_path, stats_path):
    """calib(case) -> (peaks, stats) for that case's fold, ready for CompetitiveAssigner.

    folds_path: {"k", "seed", "folds": {case: fold}, "always_fit": [case, ...]}
    (written once, before the run, as part of the pre-registration).
    stats_path: scripts/fit_organ_stats.py output over all 55 patients
    (--include_single_level), carrying per_case and per_case_moments.
    A case missing from the folds raises KeyError: scoring it would need
    constants from a fold it was never assigned to.
    """
    spec = json.loads(Path(folds_path).read_text())
    st = json.loads(Path(stats_path).read_text())
    folds, always = spec["folds"], spec["always_fit"]
    cache = {}

    def calib(case):
        fold = folds[case]
        if fold not in cache:
            cache[fold] = (fold_peaks(st["per_case"], folds, always, fold),
                           fold_stats(st["per_case_moments"], folds, always, fold))
        return cache[fold]
    return calib


def folds_spec(eval_cases, all_cases, k, seed):
    """The folds file: evaluated patients split into k folds, everyone else
    (the no-lesion patients) in always_fit."""
    ev = sorted(set(eval_cases))
    return {"k": k, "seed": seed, "folds": make_folds(ev, k, seed),
            "always_fit": sorted(set(all_cases) - set(ev))}
