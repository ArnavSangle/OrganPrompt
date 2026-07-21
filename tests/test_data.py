import pytest
import numpy as np
from src.data.organmnist import load_organamnist
from src.data.dataset import ImageTextDataset
from src.labels.templates import ZERO_SHOT_TEMPLATES

def test_train_split_shape():
    train, val, test = load_organamnist(size=28)
    img, label = train[0]
    assert img.shape == (1, 28, 28), f"Expected (1,28,28), got {img.shape}"

def test_correct_num_classes():
    train, _, _ = load_organamnist(size=28)
    labels = [int(train[i][1][0]) for i in range(len(train))]
    assert len(set(labels)) == 11, f"Expected 11 classes, got {len(set(labels))}"

def test_224_resize():
    train, _, _ = load_organamnist(size=224)
    img, _ = train[0]
    assert img.shape == (3, 224, 224), f"Expected (3,224,224), got {img.shape}"

def test_split_sizes():
    train, val, test = load_organamnist(size=28)
    assert len(train) == 34561
    assert len(val) == 6491
    assert len(test) == 17778

def test_image_text_dataset_item():
    train, _, _ = load_organamnist(size=224)
    captions = {i: [ZERO_SHOT_TEMPLATES[i]] for i in range(11)}
    ds = ImageTextDataset(train, captions)
    img, caption, label = ds[0]
    assert img.shape == (3, 224, 224)
    assert isinstance(caption, str)
    assert 0 <= label <= 10

def test_image_text_dataset_length():
    train, _, _ = load_organamnist(size=224)
    captions = {i: [ZERO_SHOT_TEMPLATES[i]] for i in range(11)}
    ds = ImageTextDataset(train, captions)
    assert len(ds) == len(train)
