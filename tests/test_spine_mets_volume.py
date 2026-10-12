from pathlib import Path
import numpy as np
import pytest
from src.data.spine_mets import find_series, load_ct_volume, load_vertebra_masks

ROOT = Path("data/Spine-Mets-CT-SEG_v1_2024/spine_mets_ct_seg")


@pytest.mark.skipif(not ROOT.exists(), reason="Spine-Mets not present")
def test_ct_volume_and_masks_align():
    case = sorted(p for p in ROOT.iterdir() if p.is_dir())[0]
    ct, seg = find_series(str(case))
    vol, spacing = load_ct_volume(ct)
    assert vol.ndim == 3 and vol.shape[1:] == (512, 512)
    assert vol.min() < -500 and vol.max() > 500      # real HU range
    assert all(s > 0 for s in spacing) and len(spacing) == 3
    masks = load_vertebra_masks(seg, ct)
    assert len(masks) >= 5                            # several vertebrae
    any_level = next(iter(masks))
    assert masks[any_level].shape == vol.shape        # aligned to CT grid
    assert masks[any_level].dtype == bool
    assert masks[any_level].sum() > 0


# --- SEG -> CT in-plane geometry (BSPC revision, Reviewer 5 point 3) ---------
# Spine-Mets SEGs store their column direction as (0,-1,0) against the CT's
# (0,1,0); pasting SEG frames onto the CT grid by z alone mirrored every
# vertebra anterior-posterior. These pin the geometric mapping that replaced it.
from src.data.spine_mets import _resample_seg_frame

IDENT = (1, 0, 0, 0, 1, 0)


def test_flipped_row_direction_is_unflipped():
    # CT rows run +y from y=97; SEG rows run -y from y=100 (the opposite end).
    frame = np.zeros((4, 3), bool)
    frame[0, 1] = True                                  # SEG row 0 -> y=100
    out = _resample_seg_frame(
        frame, seg_iop=(1, 0, 0, 0, -1, 0), seg_ipp=(0.0, 100.0), seg_ps=(1.0, 1.0),
        ct_iop=IDENT, ct_ipp=(0.0, 97.0), ct_ps=(1.0, 1.0), ct_shape=(4, 3))
    assert out[3, 1] and out.sum() == 1                  # CT's last row


def test_matching_geometry_is_identity():
    frame = np.random.default_rng(0).random((5, 5)) > 0.5
    out = _resample_seg_frame(frame, IDENT, (0.0, 0.0), (1.0, 1.0),
                              IDENT, (0.0, 0.0), (1.0, 1.0), (5, 5))
    assert (out == frame).all()


def test_offset_origin_shifts_mask():
    frame = np.zeros((5, 5), bool)
    frame[0, 0] = True
    out = _resample_seg_frame(frame, IDENT, (2.0, 1.0), (1.0, 1.0),
                              IDENT, (0.0, 0.0), (1.0, 1.0), (5, 5))
    assert out[1, 2] and out.sum() == 1                  # +2 mm x -> col, +1 mm y -> row


def test_anisotropic_spacing_uses_row_col_order():
    # DICOM PixelSpacing is (row spacing, col spacing). SEG col step 2 mm, CT 1 mm.
    frame = np.zeros((3, 3), bool)
    frame[0, 1] = True                                  # x = 2 mm
    out = _resample_seg_frame(frame, IDENT, (0.0, 0.0), (1.0, 2.0),
                              IDENT, (0.0, 0.0), (1.0, 1.0), (3, 3))
    assert out[0, 2] and out.sum() == 1


def test_non_axis_aligned_raises():
    with pytest.raises(ValueError):
        _resample_seg_frame(np.ones((2, 2), bool), (0.9, 0.1, 0, 0, 1, 0), (0.0, 0.0),
                            (1.0, 1.0), IDENT, (0.0, 0.0), (1.0, 1.0), (2, 2))


def test_pixels_outside_ct_grid_raise():
    frame = np.zeros((2, 2), bool)
    frame[1, 1] = True
    with pytest.raises(ValueError):
        _resample_seg_frame(frame, IDENT, (5.0, 5.0), (1.0, 1.0),
                            IDENT, (0.0, 0.0), (1.0, 1.0), (2, 2))


@pytest.mark.skipif(not ROOT.exists(), reason="Spine-Mets not present")
@pytest.mark.parametrize("case", ["10355", "13683"])
def test_vertebra_masks_sit_on_bone(case):
    ct, seg = find_series(str(ROOT / case))
    vol, _ = load_ct_volume(ct)
    for lvl, m in load_vertebra_masks(seg, ct).items():
        assert vol[m].mean() > 150, f"{case}:{lvl} mean HU {vol[m].mean():.0f}"
