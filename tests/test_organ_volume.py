import numpy as np
import torch
from src.tumor import organ_volume as ov


class _FakeExplainer:
    def __init__(self, model, device="cpu"): pass
    def generate(self, image, text_features, target_class, image_size=224):
        # deterministic: constant map equal to class index / 10
        return np.full((image_size, image_size), target_class / 10.0, np.float32)
    def remove_hooks(self): pass


def test_build_organ_volume_shape_and_downsample(monkeypatch):
    monkeypatch.setattr(ov, "BioMedCLIPPatchSimilarity", _FakeExplainer)
    ct = np.random.uniform(-500, 500, size=(4, 512, 512)).astype(np.float32)
    tf = torch.zeros(11, 512)
    vol = ov.build_organ_volume(ct, model=object(), text_features=tf,
                                device="cpu", organs=[6, 7])
    assert vol.shape == (2, 4, 512, 512)          # 2 organs, D=4, native H,W
    assert np.allclose(vol[0], 6 / 10.0)          # organ 6 constant map
    assert np.allclose(vol[1], 7 / 10.0)
