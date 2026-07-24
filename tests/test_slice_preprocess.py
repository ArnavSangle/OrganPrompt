import numpy as np
import torch
from src.tumor.organ_volume import preprocess_slice


def test_preprocess_slice_shape_and_range():
    hu = np.random.uniform(-1000, 1000, size=(512, 512)).astype(np.float32)
    t = preprocess_slice(hu, organ_class=6)      # liver -> default window
    assert isinstance(t, torch.Tensor)
    assert t.shape == (3, 224, 224)
    assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0


def test_preprocess_slice_lung_window_differs():
    hu = np.full((512, 512), -600.0, np.float32)   # lung-ish HU
    liver = preprocess_slice(hu, 6)                # window (-150,250) -> clipped low
    lung = preprocess_slice(hu, 7)                 # window (-1000,-200) -> mid-range
    assert float(lung.mean()) > float(liver.mean())
