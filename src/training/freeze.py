UNFREEZE_PATTERNS = [
    "visual.trunk.blocks.8",
    "visual.trunk.blocks.9",
    "visual.trunk.blocks.10",
    "visual.trunk.blocks.11",
    "text.transformer.encoder.layer.8",
    "text.transformer.encoder.layer.9",
    "text.transformer.encoder.layer.10",
    "text.transformer.encoder.layer.11",
    "text.proj",
    "logit_scale",
    "logit_bias",
]

def freeze_model(model) -> None:
    """Freeze all BioMedCLIP params, then selectively unfreeze by name pattern.

    Unfreezes last 4 ViT image blocks, last 4 BERT text layers,
    text projection, logit_scale, and logit_bias (~57M of 195M params).
    Modifies model in-place.
    """
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if any(pattern in name for pattern in UNFREEZE_PATTERNS):
            param.requires_grad = True
