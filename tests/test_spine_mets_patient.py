from pathlib import Path
import numpy as np
import pytest
from src.data.spine_mets import load_patient
from src.data.spine_mets_labels import load_lesion_labels

ROOT = Path("data/Spine-Mets-CT-SEG_v1_2024")
CASES = ROOT / "spine_mets_ct_seg"


@pytest.mark.skipif(not (ROOT / "Spine-Mets-CT-SEG_Clinical.xlsx").exists(), reason="data absent")
def test_load_patient_builds_tumor_mask():
    labels = load_lesion_labels(str(ROOT / "Spine-Mets-CT-SEG_Clinical.xlsx"))
    # pick a case that (a) exists on disk and (b) has >=1 labeled level
    case = next(c.name for c in sorted(CASES.iterdir())
                if c.is_dir() and labels.get(c.name))
    p = load_patient(str(ROOT), case, labels)
    assert p.ct.shape == p.tumor_mask.shape == p.lesion_type.shape
    assert p.tumor_mask.sum() > 0
    assert set(np.unique(p.lesion_type)).issubset({0, 1, 2, 3})
    assert p.lesion_type[p.tumor_mask].min() >= 1     # labeled where tumor is
