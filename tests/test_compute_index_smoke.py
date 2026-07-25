import json
from pathlib import Path

import numpy as np
import pytest

import scripts.compute_index as ci

ROOT = Path("data/Spine-Mets-CT-SEG_v1_2024")


@pytest.mark.skipif(not (ROOT / "Spine-Mets-CT-SEG_Clinical.xlsx").exists(), reason="data absent")
@pytest.mark.parametrize("assign", ["competitive", "independent"])
def test_compute_index_writes_json(tmp_path, monkeypatch, assign):
    # stub the model + organ volume so no GPU/network is needed
    monkeypatch.setattr(ci, "load_organcam", lambda ckpt, device: (object(), None))

    def fake_iter(ct, model, tf, device, organs=None):
        organs = organs or list(range(11))
        rng = np.random.RandomState(0)
        for oc in organs:
            v = rng.uniform(-1.0, -0.5, size=ct.shape).astype(np.float32)
            # give each organ its own distinct territory so competitive assignment
            # has something to separate, plus a clear peak inside it. Row band is
            # at cropped z=0, which is guaranteed tumor-free (the crop starts
            # limit_slices before the first true tumor slice).
            r0 = 2 + oc * 3
            v[0, r0:r0 + 3, 0:5] = 1.0
            yield oc, v

    monkeypatch.setattr(ci, "iter_organ_similarity", fake_iter)
    out = ci.run(root=str(ROOT), case=None, checkpoint="x", device="cpu",
                 alpha=0.5, beta=0.5, limit_slices=8, out_dir=str(tmp_path),
                 assign=assign)
    data = json.loads(Path(out).read_text())
    assert "index" in data and "per_organ" in data
    assert data["tumor_voxels"] > 0
    assert "organ_sizes" in data
    assert data["assign"] == assign
    assert len(data["per_organ"]) == 11
    # every organ record carries a directly-readable distance
    for v in data["per_organ"].values():
        assert "distance_mm" in v and "voxels" in v


@pytest.mark.skipif(not (ROOT / "Spine-Mets-CT-SEG_Clinical.xlsx").exists(), reason="data absent")
def test_competitive_reports_absent_organs_independent_does_not(tmp_path, monkeypatch):
    """The behavioural difference that motivates competition.

    One organ dominates everywhere. Under independent thresholding all 11 organs
    still get a mask; under competition the losers are reported absent.
    """
    monkeypatch.setattr(ci, "load_organcam", lambda ckpt, device: (object(), None))

    def fake_iter(ct, model, tf, device, organs=None):
        organs = organs or list(range(11))
        rng = np.random.RandomState(0)
        for oc in organs:
            # genuine variation, so the independent path's percentile threshold is
            # meaningful and every organ really does get a component
            lo, hi = (0.89, 0.90) if oc == 7 else (0.0, 0.02)   # lung-left dominates
            yield oc, rng.uniform(lo, hi, size=ct.shape).astype(np.float32)

    monkeypatch.setattr(ci, "iter_organ_similarity", fake_iter)
    kw = dict(root=str(ROOT), case=None, checkpoint="x", device="cpu",
              alpha=0.5, beta=0.5, limit_slices=8)
    comp = json.loads(Path(ci.run(**kw, out_dir=str(tmp_path / "c"),
                                  assign="competitive")).read_text())
    indep = json.loads(Path(ci.run(**kw, out_dir=str(tmp_path / "i"),
                                   assign="independent")).read_text())
    # lung-right is exempt: it is lung-left's anatomical pair, and a pair is
    # present or absent as a unit, so a dominant left lung vouches for the right
    losers = {n for n in comp["organ_sizes"] if n not in ("lung-left", "lung-right")}
    # competition strips every organ that lost its peak to lung-left ...
    assert losers <= set(comp["absent"]), comp["absent"]
    assert "lung-right" not in comp["absent"], "a pair must not be split"
    # ... while independent thresholding hands each of them a mask regardless
    assert not (losers & set(indep["absent"])), indep["absent"]
