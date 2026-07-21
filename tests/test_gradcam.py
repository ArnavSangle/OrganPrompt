import pytest
import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import open_clip
from src.explainability.gradcam import BioMedCLIPGradCAM

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


@pytest.fixture(scope="module")
def model():
    m, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)
    m.eval()
    return m


def test_gradcam_output_shape(model):
    gradcam = BioMedCLIPGradCAM(model, device="cpu")
    image = torch.randn(1, 3, 224, 224)
    text_features = F.normalize(torch.randn(11, 512), dim=-1)
    mask = gradcam.generate(image, text_features, target_class=0)
    gradcam.remove_hooks()
    assert mask.shape == (224, 224), f"Expected (224,224), got {mask.shape}"
    assert mask.dtype == np.float32


def test_gradcam_values_in_range(model):
    gradcam = BioMedCLIPGradCAM(model, device="cpu")
    image = torch.randn(1, 3, 224, 224)
    text_features = F.normalize(torch.randn(11, 512), dim=-1)
    mask = gradcam.generate(image, text_features, target_class=3)
    gradcam.remove_hooks()
    assert float(mask.min()) >= 0.0, f"min={mask.min()} < 0"
    assert float(mask.max()) <= 1.0, f"max={mask.max()} > 1"


def test_gradcam_different_classes_differ(model):
    """Same image, different target class → different gradient maps (pre-ReLU)."""
    gradcam = BioMedCLIPGradCAM(model, device="cpu")
    image = torch.randn(1, 3, 224, 224)
    text_features = F.normalize(torch.randn(11, 512), dim=-1)
    gradcam.generate(image, text_features, target_class=0)
    grads0 = gradcam._gradients.clone()
    gradcam.generate(image, text_features, target_class=5)
    grads5 = gradcam._gradients.clone()
    gradcam.remove_hooks()
    assert not torch.allclose(grads0, grads5), "Gradient maps for different classes should differ"
