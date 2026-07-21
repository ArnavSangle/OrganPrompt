"""Fine-tune BioMedCLIP on OrganAMNIST image-text pairs.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 20 --batch_size 64 --lr 5e-5
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

import open_clip
from open_clip import get_tokenizer
from src.data.organmnist import load_organamnist
from src.data.dataset import ImageTextDataset
from src.training.freeze import freeze_model
from src.training.trainer import Trainer

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CAPTIONS_PATH = Path(__file__).parent.parent / "data" / "prepared" / "captions.json"


def load_captions() -> dict[int, list[str]]:
    with open(CAPTIONS_PATH) as f:
        data = json.load(f)
    class_captions: dict[int, list[str]] = {i: [] for i in range(11)}
    for record in data["splits"]["train"]:
        label = record["label"]
        if not class_captions[label]:
            class_captions[label] = (
                [record["zero_shot"]] +
                [record["few_shot"]] +
                record["variants"]
            )
    return class_captions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print("Loading BioMedCLIP...")
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)
    model = model.to(args.device)
    freeze_model(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model on: {next(model.parameters()).device}", flush=True)
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    print("Loading tokenizer...", flush=True)
    tokenizer = get_tokenizer(MODEL_NAME)
    print("Tokenizer ready.", flush=True)

    print("Loading OrganAMNIST...", flush=True)
    train_ds_raw, val_ds_raw, _ = load_organamnist(size=224)
    print("OrganAMNIST loaded.", flush=True)
    print("Loading captions...", flush=True)
    captions = load_captions()
    print("Pre-tokenizing captions...", flush=True)
    tokenized_captions = {
        label: [tokenizer([cap])[0] for cap in caps]
        for label, caps in captions.items()
    }
    print("Captions ready.", flush=True)

    train_ds = ImageTextDataset(train_ds_raw, tokenized_captions)
    val_ds = ImageTextDataset(val_ds_raw, tokenized_captions)
    print("Datasets wrapped.", flush=True)

    pin_memory = args.device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=pin_memory)
    print("DataLoaders ready.", flush=True)

    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")
    print(f"Starting training: {args.epochs} epochs, lr={args.lr}, batch={args.batch_size}", flush=True)

    # Verify GPU is actually being used
    if args.device == "cuda":
        images_sample, tokens_sample, _ = next(iter(train_loader))
        images_sample = images_sample.to(args.device)
        tokens_sample = tokens_sample.to(args.device)
        print(f"Image tensor device: {images_sample.device}", flush=True)
        print(f"Token tensor device: {tokens_sample.device}", flush=True)
        next_param = next(model.parameters())
        print(f"Model device: {next_param.device}", flush=True)
        del images_sample, tokens_sample

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer.train(epochs=args.epochs)
    print(f"Done. Best val loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
