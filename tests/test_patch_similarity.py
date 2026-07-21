import pytest
import torch
import torch.nn.functional as F
import numpy as np
import open_clip
from src.explainability.patch_similarity import BioMedCLIPPatchSimilarity

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


@pytest.fixture(scope="module")
def model():
    m, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)
    m.eval()
    return m


def test_patch_sim_output_shape(model):
    ps = BioMedCLIPPatchSimilarity(model, device="cpu")
    image = torch.randn(1, 3, 224, 224)
    text_features = F.normalize(torch.randn(11, 512), dim=-1)
    mask = ps.generate(image, text_features, target_class=0)
    ps.remove_hooks()
    assert mask.shape == (224, 224)
    assert mask.dtype == np.float32


def test_patch_sim_values_in_range(model):
    ps = BioMedCLIPPatchSimilarity(model, device="cpu")
    image = torch.randn(1, 3, 224, 224)
    text_features = F.normalize(torch.randn(11, 512), dim=-1)
    mask = ps.generate(image, text_features, target_class=3)
    ps.remove_hooks()
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0


def test_patch_sim_four_hooks_registered(model):
    ps = BioMedCLIPPatchSimilarity(model, device="cpu")
    assert len(ps._hooks) == 4
    ps.remove_hooks()
    assert len(ps._hooks) == 0


def test_patch_sim_different_classes_differ(model):
    ps = BioMedCLIPPatchSimilarity(model, device="cpu")
    image = torch.randn(1, 3, 224, 224)
    text_features = F.normalize(torch.randn(11, 512), dim=-1)
    mask0 = ps.generate(image, text_features, target_class=0)
    mask5 = ps.generate(image, text_features, target_class=5)
    ps.remove_hooks()
    assert not np.allclose(mask0, mask5), "Masks for different classes should differ"


def test_patch_sim_with_negatives(model):
    ps = BioMedCLIPPatchSimilarity(model, device="cpu")
    image = torch.randn(1, 3, 224, 224)
    text_features = F.normalize(torch.randn(11, 512), dim=-1)
    neg_features  = F.normalize(torch.randn(7, 512), dim=-1)
    mask = ps.generate(image, text_features, target_class=0, neg_text_features=neg_features)
    ps.remove_hooks()
    assert mask.shape == (224, 224)
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0
