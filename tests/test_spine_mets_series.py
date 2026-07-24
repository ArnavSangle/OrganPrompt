from pathlib import Path
import pytest
from src.data.spine_mets import find_series

ROOT = Path("data/Spine-Mets-CT-SEG_v1_2024/spine_mets_ct_seg")


@pytest.mark.skipif(not ROOT.exists(), reason="Spine-Mets not present")
def test_find_series_returns_ct_and_seg():
    case = sorted(p for p in ROOT.iterdir() if p.is_dir())[0]
    ct, seg = find_series(str(case))
    assert len(ct) > 100                       # a full CT stack
    assert seg.endswith(".dcm")
    import pydicom
    assert pydicom.dcmread(seg, stop_before_pixels=True).Modality == "SEG"
    # CT sorted ascending by z
    zs = [float(pydicom.dcmread(p, stop_before_pixels=True).ImagePositionPatient[2]) for p in ct[:5]]
    assert zs == sorted(zs)
