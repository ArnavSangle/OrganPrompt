"""Fine-tune BioMedCLIP on TotalSegmentator axial CT slices (text-supervised).

Usage:
    python scripts/train.py --root Totalsegmentator_dataset_v201
    python scripts/train.py --root Totalsegmentator_dataset_v201 --epochs 20 --batch_size 32 --lr 1e-4
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

import open_clip
from open_clip import get_tokenizer
from src.data.totalsegmentator import load_totalsegmentator
from src.data.dataset import ImageTextDataset
from src.training.freeze import freeze_model
from src.training.trainer import Trainer
from src.labels.templates import FEW_SHOT_TEMPLATES

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                        help="Path to Totalsegmentator_dataset_v201/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/totalseg")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Slices per organ per volume (default 5)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes (default 4; set 0 if errors)")
    parser.add_argument("--lambda_patch", type=float, default=0.5,
                        help="Weight for patch-level contrastive loss (0 = disabled, default 0.5)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print("Loading BioMedCLIP (base weights — no checkpoint)...")
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)
    model = model.to(args.device)
    freeze_model(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    print("Loading tokenizer...")
    tokenizer = get_tokenizer(MODEL_NAME)

    print("Pre-tokenizing FEW_SHOT_TEMPLATES...")
    tokenized_captions = {
        label: [tokenizer([cap])[0] for cap in [caption]]
        for label, caption in FEW_SHOT_TEMPLATES.items()
    }

    include_masks = args.lambda_patch > 0
    print(f"Loading TotalSegmentator from {args.root} (top_k={args.top_k}, masks={include_masks})...")
    train_ds_raw, val_ds_raw, _ = load_totalsegmentator(
        args.root, top_k=args.top_k, include_masks=include_masks
    )

    train_ds = ImageTextDataset(train_ds_raw, tokenized_captions)
    val_ds   = ImageTextDataset(val_ds_raw,   tokenized_captions)
    print(f"Train: {len(train_ds)} slices | Val: {len(val_ds)} slices")

    pin_memory = args.device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=pin_memory, persistent_workers=args.num_workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=pin_memory, persistent_workers=args.num_workers > 0)

    if args.device == "cuda":
        images_s, tokens_s, *_ = next(iter(train_loader))
        print(f"Image device check: {images_s.to(args.device).device}")
        del images_s, tokens_s

    print(f"lambda_patch={args.lambda_patch} ({'enabled' if args.lambda_patch > 0 else 'disabled'})")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=args.checkpoint_dir,
        lambda_patch=args.lambda_patch,
    )
    trainer.train(epochs=args.epochs)
    if trainer.patch_loss_fn is not None:
        trainer.patch_loss_fn.remove_hook()
    print(f"Done. Best val loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
