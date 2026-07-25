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


def test_patch_sim_returns_raw_unnormalised_similarity(model):
    """The map is raw cosine similarity, NOT rescaled to [0, 1].

    Per-image min-max normalisation would force every slice to span the full
    range, destroying the cross-image comparability that every downstream
    threshold depends on -- a slice containing no instance of the organ would
    be stretched to look exactly like one that does. So the only bound is the
    mathematical one: both operands are L2-normalised, so the dot product
    cannot leave [-1, 1].
    """
    ps = BioMedCLIPPatchSimilarity(model, device="cpu")
    image = torch.randn(1, 3, 224, 224)
    text_features = F.normalize(torch.randn(11, 512), dim=-1)
    mask = ps.generate(image, text_features, target_class=3)
    ps.remove_hooks()
    assert float(mask.min()) >= -1.0
    assert float(mask.max()) <= 1.0
    # not min-max stretched: an untrained/random alignment must not span [0, 1]
    assert float(mask.max()) - float(mask.min()) < 1.0


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
    # difference of two cosine similarities -> [-2, 2], and routinely negative
    # wherever a hard negative outscores the target class
    assert float(mask.min()) >= -2.0
    assert float(mask.max()) <= 2.0


def test_patch_sim_subtracts_the_strongest_negative(model):
    """A negative that ties the target must cancel it exactly, and adding
    further negatives must not undo that -- i.e. the subtraction is a MAX over
    negatives, not a mean or a min.

    Note this cannot be tested with random negatives: max(neg_sim) is itself
    often negative, and subtracting a negative RAISES the score. Feeding the
    target prompt back in as a negative is the one case where the expected
    result is pinned exactly.
    """
    ps = BioMedCLIPPatchSimilarity(model, device="cpu")
    image = torch.randn(1, 3, 224, 224)
    text_features = F.normalize(torch.randn(11, 512), dim=-1)
    # the target itself as sole negative -> sim - sim == 0 everywhere
    tied = ps.generate(image, text_features, target_class=0,
                       neg_text_features=text_features[0:1])
    # a max() keeps the tie dominant, so extra weaker negatives change nothing;
    # a mean() would let them pull the map away from zero
    padded = ps.generate(image, text_features, target_class=0,
                         neg_text_features=torch.cat(
                             [text_features[0:1], F.normalize(torch.randn(5, 512), dim=-1)]))
    ps.remove_hooks()
    assert np.allclose(tied, 0.0, atol=1e-5)
    assert np.all(padded <= 1e-5)
