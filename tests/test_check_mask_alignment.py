from scripts.check_mask_alignment import failures


def test_failures_flags_only_levels_below_threshold():
    audit = {
        "a": {"mean_hu": {"T1": 300.0, "T2": -120.0}},
        "b": {"mean_hu": {"L1": 149.9}},
        "c": {"error": "boom"},                     # errors are counted elsewhere
    }
    assert failures(audit) == [("a", "T2", -120.0), ("b", "L1", 149.9)]


def test_failures_empty_when_all_bone():
    assert failures({"a": {"mean_hu": {"T1": 400.0}}}) == []
