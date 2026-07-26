import numpy as np
import pytest

from src.data.scan_extent import (
    level_rank, lowest_level, caudal_margin_mm, LEVEL_ORDER,
)
import scripts.scan_extent as se


def test_level_rank_orders_cervical_above_sacral():
    """Craniocaudal order, not alphabetical.

    Sorting level strings lexically puts L1 before T1, which would report a
    thoracic-only scan as reaching the lumbar spine and invert the entire
    question this diagnostic exists to answer.
    """
    assert level_rank("C1") < level_rank("C7") < level_rank("T1")
    assert level_rank("T12") < level_rank("L1") < level_rank("L5") < level_rank("S1")
    assert LEVEL_ORDER.index("T2") == LEVEL_ORDER.index("T1") + 1


def test_lowest_level_picks_the_most_caudal_vertebra():
    assert lowest_level(["T12", "C7", "L3", "T1"]) == "L3"
    assert lowest_level(["S1", "L5"]) == "S1"
    assert lowest_level([]) is None


def test_lowest_level_ignores_unrecognised_labels():
    """A SEG can carry non-vertebral segments; they must not become the answer."""
    assert lowest_level(["T4", "spinal canal", "L1"]) == "L1"


def test_caudal_margin_is_scan_floor_to_lowest_vertebra():
    """How much imaged anatomy sits below the last segmented vertebra.

    The SEG covers only the vertebrae someone annotated, so its extent is a
    LOWER bound on what the scan contains. The unlabelled millimetres below it
    are exactly what decides whether a pelvis is in view.
    """
    vert_z = {"T12": (-300.0, -270.0), "L1": (-340.0, -302.0)}
    # scan runs from -700 (most caudal) to 0; L1 bottoms out at -340
    assert caudal_margin_mm(vert_z, z_min=-700.0) == pytest.approx(360.0)


def test_caudal_margin_is_zero_when_scan_ends_at_the_last_vertebra():
    vert_z = {"L5": (-500.0, -460.0)}
    assert caudal_margin_mm(vert_z, z_min=-500.0) == pytest.approx(0.0)


def test_caudal_margin_is_none_without_vertebrae():
    assert caudal_margin_mm({}, z_min=-500.0) is None


def _lesion(case, level, ranking, dists, expected):
    return {"case": case, "level": level, "expected": expected,
            "rank": None, "ranking": ranking,
            "distances_mm": dists, "absent": []}


def test_phantom_lesions_selects_close_impossible_bladder_and_femur():
    """The 24 detections this diagnostic is about.

    A bladder top-ranked at a thoracic level from 400mm away would be a scoring
    artifact and is not interesting; one top-ranked from 12mm away means a
    bladder-labelled mask is sitting in the chest, which is.
    """
    per = {
        "a:T4": _lesion("a", "T4", ["bladder", "lung-left"],
                        {"bladder": 12.0, "lung-left": 30.0},
                        ["lung-left", "lung-right"]),
        "b:T6": _lesion("b", "T6", ["bladder", "lung-left"],
                        {"bladder": 400.0, "lung-left": 30.0},
                        ["lung-left", "lung-right"]),
        "c:T4": _lesion("c", "T4", ["lung-right", "bladder"],
                        {"lung-right": 8.0, "bladder": 20.0},
                        ["lung-left", "lung-right"]),
        "d:L4": _lesion("d", "L4", ["femur-left", "kidney-left"],
                        {"femur-left": 15.0, "kidney-left": 40.0},
                        ["kidney-left", "kidney-right"]),
    }
    got = se.phantom_lesions(per)
    assert [g["key"] for g in got] == ["a:T4", "d:L4"]
    assert got[0]["organ"] == "bladder" and got[0]["family"] == "bladder"
    assert got[1]["family"] == "femur"


def test_phantom_lesions_can_restrict_to_thoracic_levels():
    """Thoracic is where a bladder or femur cannot be, whatever the scan shows."""
    per = {
        "a:T4": _lesion("a", "T4", ["bladder"], {"bladder": 12.0}, ["lung-left"]),
        "d:L4": _lesion("d", "L4", ["femur-left"], {"femur-left": 15.0},
                        ["kidney-left"]),
    }
    got = se.phantom_lesions(per, thoracic_only=True)
    assert [g["key"] for g in got] == ["a:T4"]


def test_phantom_lesions_ignores_expected_organs():
    """A femur top-ranked at L5, where it IS expected, is a correct detection."""
    per = {"e:L5": _lesion("e", "L5", ["femur-left"], {"femur-left": 15.0},
                           ["femur-left", "femur-right", "bladder"])}
    assert se.phantom_lesions(per) == []
