"""Figure helpers updated for the BSPC revision."""
import numpy as np
import pytest

from scripts.make_lesion_figures import chance_rank_pmf, rank_histogram
from scripts.make_qualitative_figure import case_from_summary


def test_chance_pmf_is_a_distribution_with_the_known_mean():
    for k in (1, 2, 3, 5, 7):
        p = chance_rank_pmf(k, n=11)
        assert len(p) == 11 and p.sum() == pytest.approx(1.0)
        assert (np.arange(1, 12) * p).sum() == pytest.approx(12 / (k + 1))


def test_chance_pmf_k1_is_uniform():
    assert np.allclose(chance_rank_pmf(1, n=11), np.full(11, 1 / 11))


def test_rank_histogram_counts_ranked_lesions_and_averages_chance():
    per = {"a": {"rank": 1, "expected": ["x"]},
           "b": {"rank": 3, "expected": ["x", "y"]},
           "c": {"rank": None, "expected": ["x"]}}
    observed, chance = rank_histogram(per, n=11)
    assert observed[0] == 1 and observed[2] == 1 and observed.sum() == 2
    expect = chance_rank_pmf(1, 11) + chance_rank_pmf(2, 11)   # sums, ranked lesions
    assert np.allclose(chance, expect)


def test_case_from_summary_reads_outcome_instead_of_hardcoding_it():
    summary = {"per_lesion": {"13683:T4": {
        "case": "13683", "level": "T4", "rank": 2,
        "expected": ["heart", "lung-left"], "ranking": ["liver", "heart"],
        "distances_mm": {"liver": 3.04, "heart": 7.9}}}}
    info = case_from_summary(summary, "13683", "T4")
    assert info["top_organ"] == "liver"
    assert info["distance_mm"] == pytest.approx(3.04)
    assert info["correct"] is False
    assert info["expected"] == "heart / lung-left"


def test_phantom_case_is_first_femur_top1_in_case_order():
    from scripts.make_qualitative_figure import pick_phantom_case
    per = {
        "13089:T2": {"case": "13089", "level": "T2", "ranking": ["femur-left"], "expected": ["lung-left"]},
        "10543:T8": {"case": "10543", "level": "T8", "ranking": ["femur-right"], "expected": ["lung-left"]},
        "10250:T2": {"case": "10250", "level": "T2", "ranking": ["lung-left"], "expected": ["lung-left"]},
        "10100:L5": {"case": "10100", "level": "L5", "ranking": ["femur-left"], "expected": ["femur-left"]},
    }
    assert pick_phantom_case({"per_lesion": per}) == ("10543", "T8")


def test_peak_slice_is_the_most_populated_axial_index():
    from scripts.make_qualitative_figure import peak_slice
    coords = np.array([[2, 0, 0], [5, 1, 1], [5, 2, 2], [7, 3, 3]])
    assert peak_slice(coords) == 5


def test_nearest_pair_returns_the_closest_points_in_mm():
    from scripts.make_qualitative_figure import nearest_pair
    v = np.array([[0, 0, 0], [0, 0, 5]])
    o = np.array([[2, 0, 5], [9, 9, 9]])
    pv, po, d = nearest_pair(v, o, spacing=(3.0, 1.0, 1.0))
    assert tuple(pv) == (0, 0, 5) and tuple(po) == (2, 0, 5)
    assert d == pytest.approx(6.0)                   # 2 slices x 3 mm


def test_cached_render_saves_once_and_reloads_identically(tmp_path, monkeypatch):
    import scripts.make_qualitative_figure as qf
    calls = []
    fake = {"ct": np.zeros((4, 4)), "organ": np.eye(4, dtype=bool),
            "vertebra_outline": np.zeros((4, 4), bool), "p_vertebra": (1, 2),
            "p_organ": (3, 0), "distance_mm": 14.9, "slice_offset_mm": 6.0}
    monkeypatch.setattr(qf, "CACHE", tmp_path)
    monkeypatch.setattr(qf, "render_case", lambda *a, **k: calls.append(1) or fake)
    info = {"case": "10543", "level": "T8"}
    first = qf.cached_render(info, None, None, None)
    again = qf.cached_render(info, None, None, None)
    assert len(calls) == 1
    assert again["p_organ"] == (3, 0) and again["distance_mm"] == pytest.approx(14.9)
    assert (again["organ"] == fake["organ"]).all()
