from pathlib import Path

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
