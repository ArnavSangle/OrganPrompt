import numpy as np
import pytest

import scripts.validate_lesions as vl
from src.labels.templates import CLASS_NAMES


def test_testable_lesions_keeps_every_level_not_just_single_level_patients():
    """The whole point: a patient with 4 lesions contributes 4 evaluation points.

    The per-patient validation could only use the 20 single-level patients, one
    point each, because a multi-level patient has no single expected organ set.
    Scored per lesion, each level carries its own expectation, so multi-level
    patients become usable and the evaluation set grows about ninefold.
    """
    labels = {"a": {"T2": "lytic"},
              "b": {"T2": "lytic", "L4": "blastic", "C3": "lytic"},
              "c": {}}
    got = vl.testable_lesions(labels, {"a", "b", "c"})
    assert got == {"a": ["T2"], "b": ["L4", "T2"]}      # C3 has no expected organs


def test_testable_lesions_drops_patients_absent_from_disk():
    labels = {"a": {"T2": "lytic"}, "gone": {"L4": "lytic"}}
    assert vl.testable_lesions(labels, {"a"}) == {"a": ["T2"]}


def test_summarize_counts_lesions_and_patients_separately():
    """Both numbers have to be reported.

    Lesions within one patient share a scan and one set of organ masks, so they
    are not independent observations. Quoting n=184 as if it were 184 independent
    trials overstates the evidence; the patient count is what bounds it.
    """
    per_lesion = {
        "p1:T2": {"case": "p1", "rank": 1},
        "p1:L4": {"case": "p1", "rank": 3},
        "p2:T8": {"case": "p2", "rank": None},
        "p3:L1": {"case": "p3", "rank": 2},
    }
    s = vl.summarize(per_lesion)
    assert s["n_lesions"] == 4
    assert s["n_patients"] == 3
    assert s["n_ranked"] == 3
    assert s["top1"] == 0.25
    assert s["top3"] == 0.75
    assert s["mean_rank"] == pytest.approx((1 + 3 + 2) / 3)


def test_score_patient_scores_each_lesion_against_every_organ(monkeypatch):
    """One inference pass, every lesion scored.

    Recomputing the organ masks per lesion would multiply a multi-hour run by the
    lesion count for no new information -- the masks do not depend on which lesion
    is being scored.
    """
    shape = (4, 8, 8)
    calls = []

    def fake_iter(ct, model, tf, device, organs=None):
        calls.append(ct.shape)
        organs = list(range(len(CLASS_NAMES))) if organs is None else organs
        rng = np.random.RandomState(0)
        for oc in organs:
            v = rng.uniform(0.0, 0.02, size=ct.shape).astype(np.float32)
            v[0, oc % 8, :] = 0.9          # each organ owns a distinct row
            yield oc, v

    monkeypatch.setattr(vl, "iter_organ_similarity", fake_iter)
    coords = {"T2": np.argwhere(np.zeros(shape, bool) | _row(shape, 6)),
              "L4": np.argwhere(np.zeros(shape, bool) | _row(shape, 7))}
    per_lesion = vl.score_patient(np.zeros(shape, np.float32), coords,
                                  (1.0, 1.0, 1.0), object(), None, "cpu")
    assert len(calls) == 1, "organ masks must be computed once for the patient"
    assert set(per_lesion) == {"T2", "L4"}
    for lvl in ("T2", "L4"):
        assert set(per_lesion[lvl]) == set(CLASS_NAMES.values())
        for rec in per_lesion[lvl].values():
            assert {"overlap", "proximity", "score", "distance_mm", "voxels"} <= set(rec)


def _row(shape, r):
    m = np.zeros(shape, dtype=bool)
    m[:, r, :] = True
    return m


def test_score_patient_excludes_tumor_voxels_from_organ_masks(monkeypatch):
    """Spine-Mets lesions are vertebral bone by construction, so an organ firing
    on a lesion voxel is a model artifact. Leaving it in lets a confused
    prediction at the lesion fake perfect overlap with itself."""
    shape = (3, 6, 6)

    def fake_iter(ct, model, tf, device, organs=None):
        organs = list(range(len(CLASS_NAMES))) if organs is None else organs
        for oc in organs:
            yield oc, np.full(ct.shape, 0.9, dtype=np.float32)

    monkeypatch.setattr(vl, "iter_organ_similarity", fake_iter)
    coords = {"T2": np.argwhere(_row(shape, 3))}
    per = vl.score_patient(np.zeros(shape, np.float32), coords, (1.0, 1.0, 1.0),
                           object(), None, "cpu")
    # every organ won everything, but the lesion's own voxels are carved out, so
    # no organ can report overlap with the lesion it is being scored against
    assert all(r["overlap"] == 0.0 for r in per["T2"].values())
