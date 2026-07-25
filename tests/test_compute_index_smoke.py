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

    def fake_iter(ct, model, tf, device, organs=None):
        organs = organs or list(range(11))
        rng = np.random.RandomState(0)
        for oc in organs:
            v = rng.uniform(-1.0, -0.5, size=ct.shape).astype(np.float32)
            # clear peak region at cropped z=0, which is guaranteed tumor-free
            # (crop starts limit_slices before the first true tumor slice)
            v[0, 0:5, 0:5] = 1.0
            yield oc, v

    monkeypatch.setattr(ci, "iter_organ_similarity", fake_iter)
    out = ci.run(root=str(ROOT), case=None, checkpoint="x", device="cpu",
                 alpha=0.5, beta=0.5, limit_slices=8, out_dir=str(tmp_path))
    data = json.loads(Path(out).read_text())
    assert "index" in data and "per_organ" in data
    assert data["tumor_voxels"] > 0
    assert "organ_sizes" in data
    assert len(data["per_organ"]) == 11
