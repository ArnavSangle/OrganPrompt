"""Generate patch-text similarity pseudo-masks for the OrganAMNIST test split.

CAM = sim(patch_tokens, positive_prompt) - max(sim(patch_tokens, negative_prompts))

Usage:
    python scripts/generate_masks.py
    python scripts/generate_masks.py --limit 100          # first 100 images
    python scripts/generate_masks.py --no_negatives       # skip hard-negative suppression
    python scripts/generate_masks.py --no_postprocess     # skip morphological cleanup
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from open_clip import get_tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.organmnist import load_organamnist
from src.labels.templates import CLASS_NAMES, ZERO_SHOT_TEMPLATES, NEGATIVE_PROMPTS
from src.training.freeze import UNFREEZE_PATTERNS
from src.explainability.patch_similarity import BioMedCLIPPatchSimilarity, post_process_mask


def get_dataset():
    """Return the test split as an indexable dataset.

    Each item must be (image_tensor, label) where:
      - image_tensor: torch.Tensor of shape (3, H, W), preprocessed with
        BioMedCLIP's transforms from open_clip.create_model_and_transforms
      - label: int or 1-element array-like containing the class index

    ---- SWAP POINT ----
    Replace the body of this function when moving to a new dataset.
    The rest of the pipeline (BioMedCLIPPatchSimilarity, mask saving) is
    fully dataset-agnostic and requires no other changes.
    """
    _, _, test_ds = load_organamnist(size=224)
    return test_ds

MODEL_NAME   = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
DEFAULT_CKPT = "checkpoints/epoch_009_val3.0132.pt"
OUTPUT_DIR   = Path("data/pseudo_masks")


def load_model(checkpoint: str, device: str):
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)
    model = model.to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    missing, _ = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    frozen_missing = [k for k in missing if not any(p in k for p in UNFREEZE_PATTERNS)]
    print(f"  Loaded checkpoint (epoch {ckpt['epoch']}, val {ckpt['val_loss']:.4f})")
    print(f"  {len(frozen_missing)} frozen keys not in checkpoint (expected)")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def encode_texts(model, tokenizer, texts: list, device: str) -> torch.Tensor:
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        features = F.normalize(model.encode_text(tokens), dim=-1)
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N test images (default: all 17778)")
    parser.add_argument("--target_class", type=int, default=None,
                        help="Explain this class index for every image (default: true label)")
    parser.add_argument("--no_negatives", action="store_true",
                        help="Skip hard-negative suppression")
    parser.add_argument("--no_postprocess", action="store_true",
                        help="Skip morphological post-processing")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(__file__).parent.parent / ckpt_path

    print(f"Device: {args.device}")
    print(f"Loading model from {ckpt_path}...")
    model = load_model(str(ckpt_path), args.device)

    print("Encoding text templates...")
    tokenizer     = get_tokenizer(MODEL_NAME)
    pos_templates = [ZERO_SHOT_TEMPLATES[i] for i in range(len(CLASS_NAMES))]
    text_features = encode_texts(model, tokenizer, pos_templates, args.device)  # (11, 512)

    neg_features = None
    if not args.no_negatives:
        print(f"Encoding {len(NEGATIVE_PROMPTS)} hard-negative prompts...")
        neg_features = encode_texts(model, tokenizer, NEGATIVE_PROMPTS, args.device)

    print("Loading dataset...")
    test_ds = get_dataset()
    n = min(len(test_ds), args.limit) if args.limit else len(test_ds)
    print(f"Generating masks for {n} images...")

    patch_sim = BioMedCLIPPatchSimilarity(model, device=args.device)
    metadata  = []

    for idx in range(n):
        image, label_arr = test_ds[idx]
        label  = int(label_arr[0])
        target = args.target_class if args.target_class is not None else label

        image_tensor = image.unsqueeze(0).to(args.device)
        mask = patch_sim.generate(
            image_tensor, text_features,
            target_class=target,
            neg_text_features=neg_features,
        )

        if not args.no_postprocess:
            mask = post_process_mask(mask)

        out_path = OUTPUT_DIR / f"{idx:06d}_class{label}_target{target}.npy"
        np.save(out_path, mask)

        metadata.append({
            "idx":          idx,
            "true_label":   label,
            "class_name":   CLASS_NAMES[label],
            "target_class": target,
            "target_name":  CLASS_NAMES[target],
            "mask_path":    str(out_path),
        })

        if (idx + 1) % 500 == 0 or (idx + 1) == n:
            print(f"  {idx+1}/{n}", flush=True)

    patch_sim.remove_hooks()

    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nDone. {n} masks saved to {OUTPUT_DIR}/")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
