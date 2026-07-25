import numpy as np

import scripts.validate_anatomy as va


def test_distances_mm_inverts_proximity():
    per_organ = {"liver": {"proximity": 1.0 / 11.0}, "spleen": {"proximity": 0.5}}
    d = va.distances_mm(per_organ)
    assert np.isclose(d["liver"], 10.0)
    assert np.isclose(d["spleen"], 1.0)


def test_distances_mm_zero_proximity_is_infinite():
    assert va.distances_mm({"bladder": {"proximity": 0.0}})["bladder"] == float("inf")


def test_distances_mm_round_trips_the_observed_t2_case():
    # regression on the real 10250 numbers: kidney-right ranked first at ~31mm
    # from a T2 vertebra, which is anatomically impossible and is exactly what
    # rank-only scoring failed to catch.
    d = va.distances_mm({"kidney-right": {"proximity": 0.031023260052873244}})
    assert 31.0 < d["kidney-right"] < 31.5
