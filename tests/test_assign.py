import numpy as np
import pytest

from src.tumor.assign import TYPICAL_PEAK, CompetitiveAssigner
from src.labels.templates import CLASS_NAMES

_IDX = {v: k for k, v in CLASS_NAMES.items()}
BLADDER, LUNG_L, FEMUR_L = _IDX["bladder"], _IDX["lung-left"], _IDX["femur-left"]
KID_L, KID_R, LIVER = _IDX["kidney-left"], _IDX["kidney-right"], _IDX["liver"]
_ALL1 = {n: 1.0 for n in CLASS_NAMES.values()}   # neutral calibration


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


def test_bilateral_pair_is_present_or_absent_together():
    """A scan contains both kidneys or neither -- never exactly one.

    Regression test for the failure that per-organ competition produced: the
    left/right prompts differ only by one word ("LEFT side"/"RIGHT side"), the
    model discriminates them poorly, so both peak in nearly the same place. The
    stronger one took the voxel and the weaker was declared ABSENT for a reason
    that has nothing to do with anatomy. Measured on the 20 single-level
    patients, 37% of bilateral pairs were split this way -- femur-right came back
    absent 9/20 against femur-left's 3/20.
    """
    shape = (4, 8, 8)
    blob = (slice(None), slice(2, 6), slice(None))   # spans the full x range
    kl = np.full(shape, 0.1, np.float32)
    kl[blob] = 0.9
    kr = np.full(shape, 0.1, np.float32)
    kr[blob] = 0.8          # same territory, uniformly weaker -> peak is stolen
    masks, a = _run(shape, {KID_L: kl, KID_R: kr}, _ALL1, floor=0.5)
    assert masks[KID_L].any(), "stronger side must survive"
    assert masks[KID_R].any(), "weaker side must NOT be cannibalised by its pair"
    assert not a.absent()


def test_bilateral_territory_splits_on_the_image_midline():
    """Laterality comes from position, not from the prompt.

    Every template states it explicitly and consistently: a patient-LEFT organ
    "appears on the RIGHT side of the image". So the split is a fact about the x
    axis, and reading it off geometry is strictly more reliable than trusting the
    text encoder to separate two prompts that differ by one word.
    """
    shape = (4, 8, 8)
    blob = (slice(None), slice(2, 6), slice(None))
    kl = np.full(shape, 0.1, np.float32)
    kl[blob] = 0.9
    kr = np.full(shape, 0.1, np.float32)
    kr[blob] = 0.8
    masks, _ = _run(shape, {KID_L: kl, KID_R: kr}, _ALL1, floor=0.5)
    # x >= 4 is the image's right half -> the patient's LEFT kidney
    assert masks[KID_L][:, 2:6, 4:].all()
    assert not masks[KID_L][:, :, :4].any()
    assert masks[KID_R][:, 2:6, :4].all()
    assert not masks[KID_R][:, :, 4:].any()
    assert not (masks[KID_L] & masks[KID_R]).any()      # still one voxel, one owner


def test_bilateral_group_absent_together_when_outcompeted():
    """Grouping must not make a pair unkillable -- a chest CT has no kidneys."""
    shape = (4, 8, 8)
    liver = np.full(shape, 0.95, np.float32)      # dominates everywhere
    kl = np.full(shape, 0.3, np.float32)
    kr = np.full(shape, 0.2, np.float32)
    masks, a = _run(shape, {LIVER: liver, KID_L: kl, KID_R: kr}, _ALL1, floor=0.5)
    assert masks[LIVER].any()
    assert not masks[KID_L].any() and not masks[KID_R].any()
    assert a.absent() == {KID_L, KID_R}, "a pair goes absent as a unit"


def test_bilateral_group_competes_with_its_strongest_member():
    """Presence is decided for the pair, so one confident side vouches for both.

    Here the right kidney answers strongly enough to beat the liver but the left
    does not. Anatomically that means the kidneys ARE in the scan, so both sides
    must be reported -- the pair competes as one unit, not two.
    """
    shape = (4, 8, 8)
    liver = np.full(shape, 0.5, np.float32)
    kl = np.full(shape, 0.4, np.float32)          # alone, would lose to liver
    kr = np.full(shape, 0.4, np.float32)
    kr[:, 2:6, 0:4] = 0.9                          # strong, image-left half
    masks, a = _run(shape, {LIVER: liver, KID_L: kl, KID_R: kr}, _ALL1, floor=0.5)
    assert masks[KID_R].any()
    assert not a.absent()


def test_non_bilateral_organs_are_unaffected_by_grouping():
    """Grouping is only for true anatomical pairs; liver and spleen still fight."""
    shape = (4, 8, 8)
    liver = np.full(shape, 0.9, np.float32)
    spleen = np.full(shape, 0.4, np.float32)
    masks, a = _run(shape, {LIVER: liver, _IDX["spleen"]: spleen}, _ALL1, floor=0.5)
    assert masks[LIVER].any()
    assert not masks[_IDX["spleen"]].any()
    assert a.absent() == {_IDX["spleen"]}


def test_default_typical_peaks_cover_every_class():
    assert set(TYPICAL_PEAK) == set(CLASS_NAMES.values())
    assert all(v > 0 for v in TYPICAL_PEAK.values())


def test_update_rejects_shape_mismatch():
    a = CompetitiveAssigner((2, 4, 4))
    with pytest.raises(ValueError):
        a.update(BLADDER, np.zeros((2, 4, 5), np.float32))
