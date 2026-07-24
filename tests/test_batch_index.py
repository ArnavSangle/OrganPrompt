from scripts.batch_index import aggregate_summary


def test_aggregate_summary_stats():
    s = aggregate_summary({"a": 0.2, "b": 0.8, "c": 0.5})
    assert s["n"] == 3
    assert abs(s["mean"] - 0.5) < 1e-6
    assert abs(s["median"] - 0.5) < 1e-6
    assert s["min"] == 0.2 and s["max"] == 0.8
