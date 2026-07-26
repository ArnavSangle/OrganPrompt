import scripts.validate_anatomy as va


def test_expected_organs_cover_thoracic_and_lumbar():
    assert "lung-left" in va.EXPECTED["T2"]
    assert "heart" in va.EXPECTED["T6"]
    assert "liver" in va.EXPECTED["T11"]
    assert "kidney-left" in va.EXPECTED["L3"]
    assert "bladder" in va.EXPECTED["L5"]


def test_expected_includes_lungs_at_T11_and_T12():
    # Per Clinical Radiology (PubMed 19589417, 282 PA chest radiographs), the
    # posterior costophrenic sulcus averages spinal level 13.5 +/- 0.6 -- the
    # lower half of L1 -- a full vertebral level below T12. Both T11 and T12
    # sit within (or above) that typical range, so a lung ranked highly at
    # either level is anatomically plausible, not a phantom detection. See
    # docs/references/anatomy-citations.md for full sourcing.
    assert "lung-left" in va.EXPECTED["T11"]
    assert "lung-right" in va.EXPECTED["T11"]
    assert "lung-left" in va.EXPECTED["T12"]
    assert "lung-right" in va.EXPECTED["T12"]


def test_single_level_cases_filters_to_exactly_one_level():
    labels = {"a": {"T2": "lytic"}, "b": {"T2": "lytic", "L4": "blastic"}, "c": {}}
    assert va.single_level_cases(labels) == {"a": "T2"}


def test_single_level_cases_drops_levels_with_no_expected_organ():
    labels = {"a": {"C3": "lytic"}, "b": {"L4": "lytic"}}
    assert va.single_level_cases(labels) == {"b": "L4"}


def test_rank_of_expected_returns_best_rank():
    per_organ = {
        "spleen": {"proximity": 0.9},
        "kidney-left": {"proximity": 0.5},
        "liver": {"proximity": 0.1},
    }
    # kidney-left is 2nd by proximity
    assert va.rank_of_expected(per_organ, {"kidney-left"}) == 2
    assert va.rank_of_expected(per_organ, {"spleen", "kidney-left"}) == 1


def test_rank_of_expected_missing_organ_is_none():
    assert va.rank_of_expected({"liver": {"proximity": 0.1}}, {"bladder"}) is None


def test_absent_organs_are_excluded_from_the_ranking():
    per_organ = {
        "spleen": {"proximity": 0.9},
        "bladder": {"proximity": 0.0},    # judged absent by competitive assignment
        "kidney-left": {"proximity": 0.4},
    }
    assert va.detected(per_organ) == ["spleen", "kidney-left"]
    # bladder must not be rankable at all -- "not found" is a miss, not last place
    assert va.rank_of_expected(per_organ, {"bladder"}) is None
    assert va.rank_of_expected(per_organ, {"kidney-left"}) == 2


def test_summarize_reports_topk_hit_rates():
    per_case = {"a": {"rank": 1}, "b": {"rank": 3}, "c": {"rank": 7}, "d": {"rank": None}}
    s = va.summarize(per_case)
    assert s["n"] == 4
    assert s["top1"] == 0.25
    assert s["top3"] == 0.5
    assert s["mean_rank"] == (1 + 3 + 7) / 3
