import numpy as np
from src.tumor.index import binarize_organs, denoise_organ_mask, patient_index, select_peak_component


def test_binarize_applies_per_organ_threshold():
    vol = np.zeros((2, 3, 4, 4), np.float32)
    vol[0] = 0.2; vol[1] = 0.8
    b = binarize_organs(vol, thresholds={6: 0.5, 7: 0.5}, organs=[6, 7])
    assert b[6].sum() == 0 and b[7].all()


def test_patient_index_reports_max_and_per_organ():
    shape = (3, 4, 4)
    tumor = np.zeros(shape, bool); tumor[0, 0, 0] = True
    liver = np.zeros(shape, bool); liver[0, 0, 1] = True   # 1 voxel away
    kidney = np.zeros(shape, bool); kidney[2, 3, 3] = True # far
    res = patient_index(tumor, {6: liver, 4: kidney}, spacing=(1,1,1),
                        alpha=0.5, beta=0.5)
    assert "liver" in res["per_organ"] and "kidney-left" in res["per_organ"]
    assert res["index"] == max(o["score"] for o in res["per_organ"].values())
    assert res["per_organ"]["liver"]["score"] > res["per_organ"]["kidney-left"]["score"]


def test_denoise_removes_small_components_keeps_large():
    mask = np.zeros((2, 20, 20), bool)
    mask[0, 0:10, 0:10] = True   # 100-voxel blob, should survive
    mask[0, 15, 15] = True        # isolated speckle, should be removed
    mask[1, 5, 5] = True          # speckle in a different slice, should be removed
    out = denoise_organ_mask(mask, min_size=50)
    assert out[0, 5, 5]
    assert not out[0, 15, 15]
    assert not out[1, 5, 5]
    assert out.sum() == 100


def test_denoise_empty_slice_is_noop():
    mask = np.zeros((3, 10, 10), bool)
    out = denoise_organ_mask(mask, min_size=50)
    assert not out.any()


def test_select_peak_component_discards_disconnected_blob():
    sim = np.zeros((10, 10, 10), np.float32)
    sim[0:3, 0:3, 0:3] = 0.5    # blob A: 27 voxels
    sim[0, 0, 0] = 0.9          # global peak, inside blob A
    sim[8:10, 8:10, 8:10] = 0.6  # blob B: 8 voxels, disconnected, also above threshold
    out = select_peak_component(sim, percentile=97)
    assert out[0, 0, 0]         # peak itself kept
    assert out[1, 1, 1]         # rest of blob A kept
    assert not out[8, 8, 8]     # blob B discarded even though its raw values are higher
    assert out.sum() == 27


def test_select_peak_component_all_zero_is_empty():
    sim = np.zeros((4, 4, 4), np.float32)
    out = select_peak_component(sim, percentile=99.9)
    assert not out.any()
