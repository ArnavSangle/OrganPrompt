import numpy as np
import torch
from src.tumor import organ_volume as ov
from src.labels.templates import NEGATIVE_PROMPTS


class _FakeExplainer:
    def __init__(self, model, device="cpu"): pass
    def generate(self, image, text_features, target_class, image_size=224,
                 neg_text_features=None):
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


def test_negative_prompts_include_the_three_missing_class_negatives():
    """rib/humerus/aorta are the pre-registered vision-side test of the
    missing-class hypothesis: real structures at exactly the phantom vertebral
    levels that the 11-class vocabulary has no way to rule out."""
    text = " ".join(NEGATIVE_PROMPTS).lower()
    for word in ("rib", "humerus", "aorta"):
        assert word in text, f"NEGATIVE_PROMPTS is missing a {word} negative"
    # written in the FEW_SHOT_TEMPLATES style, not bare labels
    new_prompts = [p for p in NEGATIVE_PROMPTS
                   if any(w in p.lower() for w in ("rib", "humerus", "aorta"))]
    assert len(new_prompts) == 3
    assert all(len(p) > 80 for p in new_prompts)


def test_load_organcam_with_negatives_encodes_neg_text_features(monkeypatch):
    """load_organcam must be able to produce negative text features (built
    from NEGATIVE_PROMPTS) alongside the organ text_features, so the negatives
    can be threaded into iter_organ_similarity -> generate()."""
    calls = {}

    class _FakeModel:
        def to(self, device): return self
        def eval(self): return self
        def parameters(self): return iter(())
        def load_state_dict(self, *a, **k): pass
        def encode_text(self, tokens):
            calls["n_prompts"] = calls.get("n_prompts", []) + [tokens.shape[0]]
            return torch.zeros(tokens.shape[0], 512)

    def fake_create_model_and_transforms(name):
        return _FakeModel(), None, None

    def fake_get_tokenizer(name):
        return lambda prompts: torch.zeros(len(prompts), 4, dtype=torch.long)

    monkeypatch.setattr(ov.open_clip, "create_model_and_transforms",
                        fake_create_model_and_transforms)
    monkeypatch.setattr(ov, "get_tokenizer", fake_get_tokenizer)
    monkeypatch.setattr(ov.torch, "load",
                        lambda *a, **k: {"model_state_dict": {}})

    model, tf, neg_tf = ov.load_organcam("ckpt.pt", "cpu", with_negatives=True)
    assert neg_tf is not None
    assert neg_tf.shape[0] == len(NEGATIVE_PROMPTS)
    assert tf.shape[0] == 11


def test_iter_organ_similarity_threads_neg_text_features_to_generate(monkeypatch):
    """The wiring the whole diagnostic depends on: whatever negatives
    load_organcam produces must actually reach BioMedCLIPPatchSimilarity.generate,
    not just sit unused in the caller."""
    received = []

    class _RecordingExplainer:
        def __init__(self, model, device="cpu"): pass
        def generate(self, image, text_features, target_class, image_size=224,
                     neg_text_features=None):
            received.append(neg_text_features)
            return np.zeros((image_size, image_size), np.float32)
        def remove_hooks(self): pass

    monkeypatch.setattr(ov, "BioMedCLIPPatchSimilarity", _RecordingExplainer)
    ct = np.zeros((2, 8, 8), np.float32)
    tf = torch.zeros(11, 512)
    neg_tf = torch.ones(10, 512)
    list(ov.iter_organ_similarity(ct, object(), tf, "cpu", organs=[0],
                                  neg_text_features=neg_tf))
    assert len(received) == 2   # one per slice (D=2)
    assert all(r is neg_tf for r in received)
