import numpy as np
import pytest

from src.tumor.assign import TYPICAL_PEAK, CompetitiveAssigner
from src.labels.templates import CLASS_NAMES

_IDX = {v: k for k, v in CLASS_NAMES.items()}
BLADDER, LUNG_L, FEMUR_L = _IDX["bladder"], _IDX["lung-left"], _IDX["femur-left"]


def _run(shape, sims, peaks=None, floor=0.0):
    """Feed sims to an assigner and collect the finalized masks."""
    a = CompetitiveAssigner(shape, peaks=peaks)
    for oc, sim in sims.items():
        a.update(oc, sim)
    return dict(a.finalize(floor=floor)), a


def test_absent_organ_loses_its_peak_and_gets_no_mask():
    """The whole point: an organ not in the scan should come back empty.

    Independent thresholding cannot express this -- every organ always gets its
    own top-N voxels. Under competition, an organ whose own best voxel is claimed
    by a stronger organ is reported absent.
    """
    shape = (4, 8, 8)
    peaks = {"bladder": 1.0, "lung-left": 1.0}
    bladder = np.full(shape, 0.5, np.float32)     # weak everywhere, no home
    lung = np.full(shape, 0.1, np.float32)
    lung[:, 0:4, :] = 0.9                          # strong, covers bladder's peak
    bladder[0, 0, 0] = 0.55                        # bladder's global peak, inside lung
    masks, a = _run(shape, {BLADDER: bladder, LUNG_L: lung}, peaks)
    assert not masks[BLADDER].any()
    assert masks[LUNG_L].any()
    assert BLADDER in a.absent()


def test_present_organ_keeps_its_own_territory():
    shape = (4, 8, 8)
    peaks = {"bladder": 1.0, "lung-left": 1.0}
    bladder = np.full(shape, 0.1, np.float32)
    bladder[:, 6:8, :] = 0.9                       # bladder owns the bottom rows
    lung = np.full(shape, 0.1, np.float32)
    lung[:, 0:4, :] = 0.9
    masks, a = _run(shape, {BLADDER: bladder, LUNG_L: lung}, peaks)
    assert masks[BLADDER].any() and masks[LUNG_L].any()
    assert not (masks[BLADDER] & masks[LUNG_L]).any()   # one voxel, one owner
    assert masks[BLADDER][0, 7, 0]
    assert masks[LUNG_L][0, 0, 0]


def test_calibration_flips_the_winner():
    """Raw similarity is not comparable across organs.

    A femur answering 0.05 is near its ceiling (~0.059); a lung answering 0.10 is
    well below its own (~0.175). Raw argmax hands the voxel to the lung; calibrated
    argmax correctly gives it to the femur.
    """
    shape = (2, 4, 4)
    femur = np.full(shape, 0.05, np.float32)
    lung = np.full(shape, 0.10, np.float32)
    raw, _ = _run(shape, {FEMUR_L: femur, LUNG_L: lung},
                  peaks={"femur-left": 1.0, "lung-left": 1.0})
    assert raw[LUNG_L].any() and not raw[FEMUR_L].any()

    cal, _ = _run(shape, {FEMUR_L: femur, LUNG_L: lung},
                  peaks={"femur-left": 0.059, "lung-left": 0.175})
    assert cal[FEMUR_L].any() and not cal[LUNG_L].any()


def test_mask_is_the_component_containing_the_organs_peak():
    shape = (6, 6, 6)
    peaks = {"bladder": 1.0, "lung-left": 1.0}
    bladder = np.full(shape, 0.1, np.float32)
    bladder[0:2, 0:2, 0:2] = 0.8      # blob A, 8 voxels
    bladder[0, 0, 0] = 0.95           # global peak inside blob A
    bladder[5, 5, 5] = 0.9            # disconnected blob B, also strong
    lung = np.full(shape, 0.2, np.float32)
    masks, _ = _run(shape, {BLADDER: bladder, LUNG_L: lung}, peaks, floor=0.5)
    assert masks[BLADDER][0, 0, 0]
    assert masks[BLADDER].sum() == 8          # blob A only
    assert not masks[BLADDER][5, 5, 5]        # blob B dropped as disconnected


def test_floor_suppresses_weak_winners():
    """Winning locally is not enough -- a winner must still answer near its own
    typical peak, or every voxel in the body would be assigned to something."""
    shape = (3, 4, 4)
    peaks = {"bladder": 1.0, "lung-left": 1.0}
    bladder = np.full(shape, 0.9, np.float32)
    lung = np.full(shape, 0.05, np.float32)   # lung wins nothing
    masks, _ = _run(shape, {BLADDER: bladder, LUNG_L: lung}, peaks, floor=0.6)
    assert masks[BLADDER].all()

    weak = np.full(shape, 0.3, np.float32)    # below the 0.6 floor everywhere
    masks2, _ = _run(shape, {BLADDER: weak, LUNG_L: lung}, peaks, floor=0.6)
    assert not masks2[BLADDER].any()


def test_result_is_independent_of_streaming_order():
    shape = (4, 6, 6)
    peaks = {"bladder": 1.0, "lung-left": 1.0, "femur-left": 1.0}
    rng = np.random.RandomState(0)
    sims = {oc: rng.uniform(0, 1, shape).astype(np.float32)
            for oc in (BLADDER, LUNG_L, FEMUR_L)}
    a, _ = _run(shape, sims, peaks, floor=0.3)
    b, _ = _run(shape, dict(reversed(list(sims.items()))), peaks, floor=0.3)
    for oc in sims:
        assert np.array_equal(a[oc], b[oc])


def test_ties_do_not_double_assign():
    shape = (2, 3, 3)
    peaks = {"bladder": 1.0, "lung-left": 1.0}
    same = np.full(shape, 0.8, np.float32)
    masks, _ = _run(shape, {BLADDER: same.copy(), LUNG_L: same.copy()}, peaks, floor=0.5)
    assert not (masks[BLADDER] & masks[LUNG_L]).any()


def test_default_typical_peaks_cover_every_class():
    assert set(TYPICAL_PEAK) == set(CLASS_NAMES.values())
    assert all(v > 0 for v in TYPICAL_PEAK.values())


def test_update_rejects_shape_mismatch():
    a = CompetitiveAssigner((2, 4, 4))
    with pytest.raises(ValueError):
        a.update(BLADDER, np.zeros((2, 4, 5), np.float32))
