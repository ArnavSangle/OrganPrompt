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
