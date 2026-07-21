import torch
import open_clip
from src.training.freeze import freeze_model

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

def _load_model():
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)
    return model

def test_total_trainable_reduced():
    model = _load_model()
    freeze_model(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert trainable < total * 0.5, f"Expected <50% trainable, got {trainable/total:.1%}"

def test_early_blocks_frozen():
    model = _load_model()
    freeze_model(model)
    for name, param in model.named_parameters():
        if "visual.trunk.blocks.0" in name or "visual.trunk.blocks.3" in name:
            assert not param.requires_grad, f"{name} should be frozen"

def test_late_blocks_unfrozen():
    model = _load_model()
    freeze_model(model)
    for name, param in model.named_parameters():
        if "visual.trunk.blocks.11" in name or "visual.trunk.blocks.8" in name:
            assert param.requires_grad, f"{name} should be trainable"

def test_text_proj_unfrozen():
    model = _load_model()
    freeze_model(model)
    for name, param in model.named_parameters():
        if "text.proj" in name:
            assert param.requires_grad, f"{name} should be trainable"

def test_logit_scale_unfrozen():
    model = _load_model()
    freeze_model(model)
    assert model.logit_scale.requires_grad
    if model.logit_bias is not None:
        assert model.logit_bias.requires_grad

import json
from torch.utils.data import DataLoader
from src.training.trainer import Trainer
from src.data.organmnist import load_organamnist
from src.labels.templates import ZERO_SHOT_TEMPLATES
from src.data.dataset import ImageTextDataset

def _make_tiny_loader():
    """8-image loader for fast smoke tests."""
    train, _, _ = load_organamnist(size=224)
    captions = {i: [ZERO_SHOT_TEMPLATES[i]] for i in range(11)}
    ds = ImageTextDataset(train, captions)
    subset = torch.utils.data.Subset(ds, list(range(8)))
    return DataLoader(subset, batch_size=4, shuffle=False)

def test_one_train_step_runs():
    model = _load_model()
    freeze_model(model)
    loader = _make_tiny_loader()
    trainer = Trainer(model, loader, loader, device="cpu")
    loss = trainer.train_epoch()
    assert isinstance(loss, float)
    assert loss > 0

def test_val_loss_runs():
    model = _load_model()
    freeze_model(model)
    loader = _make_tiny_loader()
    trainer = Trainer(model, loader, loader, device="cpu")
    val_loss = trainer.val_epoch()
    assert isinstance(val_loss, float)
    assert val_loss > 0

def test_checkpoint_saves(tmp_path):
    model = _load_model()
    freeze_model(model)
    loader = _make_tiny_loader()
    trainer = Trainer(model, loader, loader, device="cpu",
                      checkpoint_dir=str(tmp_path))
    trainer.train_epoch()
    trainer.save_checkpoint(epoch=0, val_loss=1.0)
    files = list(tmp_path.glob("*.pt"))
    assert len(files) == 1
