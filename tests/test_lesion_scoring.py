import numpy as np

from src.tumor.index import score_organ, score_organ_against_lesions


def _mask(shape, *sl):
    m = np.zeros(shape, dtype=bool)
    m[sl] = True
    return m


def test_scores_every_lesion_against_one_organ():
    """One organ, many lesions -- the shape the per-lesion evaluation needs.

    Organ masks cost a full inference pass to produce and are volume-sized, so a
    patient's lesions must all be scored against an organ while that organ is
    resident, not by recomputing it per lesion.
    """
    shape = (4, 8, 8)
    organ = _mask(shape, slice(None), slice(0, 2), slice(None))   # rows 0-1
    lesions = {
        "T5": np.argwhere(_mask(shape, slice(None), slice(4, 5), slice(0, 1))),
        "L1": np.argwhere(_mask(shape, slice(None), slice(7, 8), slice(0, 1))),
    }
    got = score_organ_against_lesions(lesions, organ, (1.0, 1.0, 1.0))
    assert set(got) == {"T5", "L1"}
    assert got["T5"]["distance_mm"] == 3.0        # row 4 to row 1
    assert got["L1"]["distance_mm"] == 6.0        # row 7 to row 1
    assert got["T5"]["proximity"] > got["L1"]["proximity"]


def test_overlap_is_measured_per_lesion():
    shape = (4, 8, 8)
    organ = _mask(shape, slice(None), slice(0, 2), slice(None))
    inside = np.argwhere(_mask(shape, slice(None), slice(0, 1), slice(0, 1)))
    half = np.argwhere(_mask(shape, slice(None), slice(1, 3), slice(0, 1)))
    got = score_organ_against_lesions({"in": inside, "half": half}, organ,
                                      (1.0, 1.0, 1.0))
    assert got["in"]["overlap"] == 1.0
    assert got["in"]["distance_mm"] == 0.0
    assert got["half"]["overlap"] == 0.5          # row 1 inside, row 2 outside


def test_matches_score_organ_called_once_per_lesion():
    """The guard on the whole optimisation.

    Building the organ KD-tree once and querying it per lesion must give exactly
    what the per-lesion path gives. If these ever diverge, every per-lesion number
    is quietly incomparable with every per-patient number already reported.
    """
    rng = np.random.RandomState(0)
    shape = (5, 9, 9)
    organ = rng.uniform(size=shape) > 0.7
    spacing = (2.5, 0.7, 0.7)                     # anisotropic, as real scans are
    masks = {f"L{i}": rng.uniform(size=shape) > 0.95 for i in range(3)}
    coords = {k: np.argwhere(m) for k, m in masks.items()}
    batch = score_organ_against_lesions(coords, organ, spacing)
    for k, m in masks.items():
        one = score_organ(m, organ, spacing)
        for field in ("overlap", "proximity", "score", "distance_mm", "voxels"):
            assert batch[k][field] == one[field], (k, field)


def test_absent_organ_gives_zero_proximity_for_every_lesion():
    """Competitive assignment returns absent organs empty; that must surface as
    'not present' for every lesion, not as a short distance to nothing."""
    shape = (3, 6, 6)
    empty = np.zeros(shape, dtype=bool)
    coords = {"T1": np.argwhere(_mask(shape, 0, 0, 0))}
    got = score_organ_against_lesions(coords, empty, (1.0, 1.0, 1.0))
    assert got["T1"]["proximity"] == 0.0
    assert got["T1"]["distance_mm"] is None
    assert got["T1"]["voxels"] == 0


def test_empty_lesion_does_not_divide_by_zero():
    shape = (3, 6, 6)
    organ = _mask(shape, slice(None), 0, 0)
    got = score_organ_against_lesions({"gone": np.empty((0, 3), dtype=int)},
                                      organ, (1.0, 1.0, 1.0))
    assert got["gone"]["overlap"] == 0.0
    assert got["gone"]["proximity"] == 0.0
