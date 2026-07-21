from src.labels.templates import ZERO_SHOT_TEMPLATES, FEW_SHOT_TEMPLATES, CLASS_NAMES

def test_all_11_classes_have_zero_shot():
    for i in range(11):
        assert i in ZERO_SHOT_TEMPLATES, f"Class {i} missing zero-shot template"
        assert isinstance(ZERO_SHOT_TEMPLATES[i], str)
        assert len(ZERO_SHOT_TEMPLATES[i]) > 20

def test_all_11_classes_have_few_shot():
    for i in range(11):
        assert i in FEW_SHOT_TEMPLATES, f"Class {i} missing few-shot template"
        assert isinstance(FEW_SHOT_TEMPLATES[i], str)
        assert len(FEW_SHOT_TEMPLATES[i]) > len(ZERO_SHOT_TEMPLATES[i])

def test_class_names_match_medmnist():
    expected = {
        0: "bladder", 1: "femur-left", 2: "femur-right", 3: "heart",
        4: "kidney-left", 5: "kidney-right", 6: "liver",
        7: "lung-left", 8: "lung-right", 9: "pancreas", 10: "spleen"
    }
    assert CLASS_NAMES == expected

import os
import pytest
from src.labels.generator import generate_caption_variants

@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No API key set")
def test_generator_returns_correct_count():
    template = "Axial CT scan showing the liver, a large solid organ in the right upper quadrant."
    variants = generate_caption_variants(template, n=3)
    assert len(variants) == 3, f"Expected 3 variants, got {len(variants)}"

@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No API key set")
def test_generator_returns_strings():
    template = "Axial CT scan showing the spleen in the left upper quadrant."
    variants = generate_caption_variants(template, n=2)
    for v in variants:
        assert isinstance(v, str)
        assert len(v) > 20

@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No API key set")
def test_generator_variants_differ_from_template():
    template = "Axial CT scan showing the heart in the mediastinum."
    variants = generate_caption_variants(template, n=2)
    for v in variants:
        assert v != template, "Variant should differ from template"
