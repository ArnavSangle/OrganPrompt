import torch
import pytest
from src.model.biomedclip import BioMedCLIP

@pytest.fixture(scope="module")
def model():
    return BioMedCLIP()

def test_model_loads(model):
    assert model is not None

def test_encode_image_shape(model):
    dummy = torch.zeros(2, 3, 224, 224)
    features = model.encode_image(dummy)
    assert features.shape[0] == 2, f"Expected batch dim 2, got {features.shape}"
    assert len(features.shape) == 2, f"Expected 2D output, got {features.shape}"

def test_encode_text_shape(model):
    texts = ["axial CT scan of the liver", "axial CT scan of the spleen"]
    features = model.encode_text(texts)
    assert features.shape[0] == 2, f"Expected batch dim 2, got {features.shape}"
    assert len(features.shape) == 2

def test_similarity_shape(model):
    images = torch.zeros(4, 3, 224, 224)
    texts = ["liver", "spleen", "heart", "kidney"]
    sim = model.similarity(images, texts)
    assert sim.shape == (4, 4), f"Expected (4, 4), got {sim.shape}"

def test_similarity_range(model):
    images = torch.zeros(2, 3, 224, 224)
    texts = ["liver CT scan", "spleen CT scan"]
    sim = model.similarity(images, texts)
    assert sim.min() >= -1.5 and sim.max() <= 1.5
