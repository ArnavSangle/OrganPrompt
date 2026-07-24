import json
from pathlib import Path

import numpy as np
import pytest

import scripts.compute_index as ci

ROOT = Path("data/Spine-Mets-CT-SEG_v1_2024")


@pytest.mark.skipif(not (ROOT / "Spine-Mets-CT-SEG_Clinical.xlsx").exists(), reason="data absent")
def test_compute_index_writes_json(tmp_path, monkeypatch):
    # stub the model + organ volume so no GPU/network is needed
    monkeypatch.setattr(ci, "load_organcam", lambda ckpt, device: (object(), None))

    def fake_vol(ct, model, tf, device, organs=None):
        organs = organs or list(range(11))
        v = np.zeros((len(organs), *ct.shape), np.float32)
        v[:] = 1.0   # every organ "present" everywhere -> nonzero index
        return v

    monkeypatch.setattr(ci, "build_organ_volume", fake_vol)
    out = ci.run(root=str(ROOT), case=None, checkpoint="x", device="cpu",
                 alpha=0.5, beta=0.5, limit_slices=8, out_dir=str(tmp_path))
    data = json.loads(Path(out).read_text())
    assert "index" in data and "per_organ" in data
    assert data["tumor_voxels"] > 0
    assert "organ_coverage" in data
    # organ mask is constant True everywhere in this fake, EXCEPT tumor voxels are
    # excluded (bone can't be organ tissue) -> overlap must be exactly 0 for every organ.
    assert all(o["overlap"] == 0.0 for o in data["per_organ"].values())
