from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.data.ct_org import load_ctorg_case

CTORG = Path("data/CT-ORG")


@pytest.mark.skipif(not CTORG.exists(), reason="CT-ORG absent")
def test_load_ctorg_case():
    vols = sorted(CTORG.glob("**/volume-*.nii.gz"))
    labs = sorted(CTORG.glob("**/labels-*.nii.gz"))
    assert vols and labs, "expected CT-ORG volume-*/labels-* NIfTI files"
    ct, lab, spacing = load_ctorg_case(str(vols[0]), str(labs[0]))
    assert ct.shape == lab.shape and len(spacing) == 3
    assert ct.min() < -500 and ct.max() > 500       # HU
    assert set(np.unique(lab)).issubset(set(range(6)))


def test_ctorg_labels_stored_as_sub_integer_floats_round_correctly(tmp_path):
    # Real CT-ORG label NIfTIs store integers as float32 just under the true value
    # (e.g. 2.9999999993, not 3.0). astype(int) truncates toward zero and silently
    # shifts every label down by one; this must round instead.
    shape = (4, 4, 3)
    raw_labels = np.zeros(shape, np.float32)
    raw_labels[0, 0, 0] = 0.9999999997671694   # true label 1 (liver), z=0
    raw_labels[0, 0, 1] = 2.999999999301508    # true label 3 (lungs), z=1
    raw_labels[0, 0, 2] = 4.999999998835847    # true label 5 (bone), z=2
    vol = np.random.uniform(-500, 500, size=shape).astype(np.float32)
    affine = np.eye(4)
    vol_path = tmp_path / "volume-0.nii.gz"
    lab_path = tmp_path / "labels-0.nii.gz"
    nib.save(nib.Nifti1Image(vol, affine), str(vol_path))
    nib.save(nib.Nifti1Image(raw_labels, affine), str(lab_path))
    _, lab, _ = load_ctorg_case(str(vol_path), str(lab_path))
    # loader moves z to axis 0 without swapping x/y, so raw[0,0,z] -> lab[z,0,0]
    assert lab[0, 0, 0] == 1
    assert lab[1, 0, 0] == 3
    assert lab[2, 0, 0] == 5
