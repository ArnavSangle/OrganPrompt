import numpy as np
from src.tumor.index import binarize_organs, patient_index


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
