"""Exploratory subgroup table for the per-lesion validation (BSPC revision, R3 Q7)."""
import pytest

import scripts.subgroup_lesions as sg


def _lesion(case, level, rank, k=2):
    return {"case": case, "level": level, "rank": rank,
            "expected": [f"o{i}" for i in range(k)]}


PER = {
    "a:T1": _lesion("a", "T1", 1), "a:T2": _lesion("a", "T2", 3),
    "b:L1": _lesion("b", "L1", 2), "c:L2": _lesion("c", "L2", None),
    "d:T5": _lesion("d", "T5", 1),
}
CLIN = {"a": {"age": 50, "sex": "F", "primary": "Lung"},
        "b": {"age": 70, "sex": "M", "primary": "Lung"},
        "c": {"age": 60, "sex": "M", "primary": "Renal"},
        "d": {"age": 80, "sex": "F", "primary": "Breast"}}
TYPES = {"a": {"T1": "lytic", "T2": "blastic"}, "b": {"L1": "lytic"},
         "c": {"L2": "mixed"}, "d": {"T5": "lytic"}}


def _row(rows, factor, group):
    return next(r for r in rows if r["factor"] == factor and r["group"] == group)


def test_every_factor_partitions_all_lesions():
    rows = sg.subgroup_table(PER, CLIN, TYPES, min_patients=2)
    for factor in ("sex", "age", "primary", "lesion_type"):
        assert sum(r["n_lesions"] for r in rows if r["factor"] == factor) == len(PER)


def test_sex_group_counts_and_mean_rank():
    r = _row(sg.subgroup_table(PER, CLIN, TYPES, min_patients=2), "sex", "F")
    assert (r["n_patients"], r["n_lesions"], r["n_ranked"]) == (2, 3, 3)
    assert r["mean_rank"] == pytest.approx((1 + 3 + 1) / 3)
    assert r["top1"] == pytest.approx(2 / 3)


def test_unranked_lesions_count_against_top1_but_not_mean_rank():
    r = _row(sg.subgroup_table(PER, CLIN, TYPES, min_patients=2), "sex", "M")
    assert r["n_lesions"] == 2 and r["n_ranked"] == 1
    assert r["mean_rank"] == 2.0
    assert r["top1"] == 0.0


def test_chance_uses_each_lesions_own_expected_set_size():
    r = _row(sg.subgroup_table(PER, CLIN, TYPES, min_patients=2), "sex", "F")
    assert r["chance"] == pytest.approx(12 / 3)          # (n+1)/(k+1), k=2


def test_age_splits_at_the_patient_median():
    rows = sg.subgroup_table(PER, CLIN, TYPES, min_patients=2)
    # median of 50,70,60,80 = 65 -> a,c low ; b,d high
    assert _row(rows, "age", "<= 65")["n_patients"] == 2
    assert _row(rows, "age", "> 65")["n_patients"] == 2


def test_small_primary_groups_are_pooled():
    rows = sg.subgroup_table(PER, CLIN, TYPES, min_patients=2)
    groups = {r["group"] for r in rows if r["factor"] == "primary"}
    assert groups == {"Lung", "Other"}


def test_lesion_type_is_per_lesion_not_per_patient():
    r = _row(sg.subgroup_table(PER, CLIN, TYPES, min_patients=2), "lesion_type", "lytic")
    assert r["n_lesions"] == 3 and r["n_patients"] == 3


def test_missing_clinical_record_raises():
    with pytest.raises(KeyError):
        sg.subgroup_table({"z:T1": _lesion("z", "T1", 1)}, CLIN, TYPES)
