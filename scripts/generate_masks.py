"""Generate saliency pseudo-masks for the TotalSegmentator test split.

Methods:
    gradcam   (default) backprop image-text similarity through ViT block 11
    patch_sim  cosine similarity between projected patch tokens and text

Usage:
    python scripts/generate_masks.py --root Totalsegmentator_dataset_v201 --checkpoint checkpoints/totalseg_patch_v4/epoch_015_val3.7378.pt --method patch_sim
    python scripts/generate_masks.py --root Totalsegmentator_dataset_v201 --method gradcam
    python scripts/generate_masks.py --root Totalsegmentator_dataset_v201 --method patch_sim --use_negatives --postprocess
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

from src.data.totalsegmentator import load_totalsegmentator
from src.labels.templates import CLASS_NAMES, FEW_SHOT_TEMPLATES
from src.training.freeze import UNFREEZE_PATTERNS
from src.explainability.patch_similarity import BioMedCLIPVanillaGradient, BioMedCLIPPatchSimilarity, post_process_mask

MODEL_NAME    = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
OUTPUT_DIR    = Path("data/totalseg_masks")
GT_MASKS_DIR  = OUTPUT_DIR / "gt_masks"


def get_dataset(root: str | Path, top_k: int = 5):
    """Return the TotalSegmentator test split.

    ---- SWAP POINT ----
    Each item must yield (image_tensor (3, H, W), label_int, gt_mask (224, 224))
    when eval_mode=True (which load_totalsegmentator sets automatically for the test split).
    Image tensors must be preprocessed with BioMedCLIP's transforms.
    """
    _, _, test_ds = load_totalsegmentator(root, top_k=top_k)
    return test_ds


def load_model(checkpoint: str | None, device: str):
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)
    model = model.to(device)
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = Path(__file__).parent.parent / ckpt_path
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        missing, _ = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        frozen_missing = [k for k in missing if not any(p in k for p in UNFREEZE_PATTERNS)]
        print(f"  Loaded checkpoint (epoch {ckpt['epoch']}, val {ckpt['val_loss']:.4f})")
        print(f"  {len(frozen_missing)} frozen keys not in checkpoint (expected)")
    else:
        print("  No checkpoint supplied — using base BioMedCLIP weights")
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
    parser.add_argument("--root", required=True,
                        help="Path to Totalsegmentator_dataset_v201/")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to fine-tuned checkpoint (default: base weights)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N test images (default: all)")
    parser.add_argument("--target_class", type=int, default=None,
                        help="Explain this class for every image (default: true label)")
    parser.add_argument("--method", choices=["gradcam", "patch_sim"], default="gradcam",
                        help="Saliency method (default: gradcam)")
    parser.add_argument("--use_negatives", action="store_true",
                        help="patch_sim only: subtract max other-class similarity "
                             "(harmful for most classes; keep off unless benchmarking)")
    parser.add_argument("--postprocess", action="store_true",
                        help="Apply morphological post-processing (assumes mask in [0,1]; "
                             "leave off when saving raw patch_sim similarities)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GT_MASKS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print("Loading model...")
    model = load_model(args.checkpoint, args.device)

    print("Encoding text templates...")
    tokenizer     = get_tokenizer(MODEL_NAME)
    pos_templates = [FEW_SHOT_TEMPLATES[i] for i in range(len(CLASS_NAMES))]
    text_features = encode_texts(model, tokenizer, pos_templates, args.device)  # (11, 512)

    print(f"Loading TotalSegmentator test split from {args.root}...")
    test_ds = get_dataset(args.root, top_k=args.top_k)
    n = min(len(test_ds), args.limit) if args.limit else len(test_ds)
    print(f"Method: {args.method} | Images: {n}")

    if args.method == "gradcam":
        explainer = BioMedCLIPVanillaGradient(model, device=args.device)
    else:
        explainer = BioMedCLIPPatchSimilarity(model, device=args.device)
    use_negatives = (args.method == "patch_sim") and args.use_negatives

    metadata = []

    for idx in range(n):
        image, label, gt_mask = test_ds[idx]
        target = args.target_class if args.target_class is not None else label

        image_tensor = image.unsqueeze(0).to(args.device)

        if args.method == "gradcam":
            mask = explainer.generate(image_tensor, text_features, target_class=target)
        else:
            neg_features = None
            if use_negatives:
                neg_idx = [i for i in range(len(CLASS_NAMES)) if i != target]
                neg_features = text_features[neg_idx]
            mask = explainer.generate(
                image_tensor, text_features,
                target_class=target,
                neg_text_features=neg_features,
            )

        if args.postprocess:
            mask = post_process_mask(mask)

        out_path    = OUTPUT_DIR / f"{idx:06d}_class{label}_target{target}.npy"
        gt_out_path = GT_MASKS_DIR / f"{idx:06d}_class{label}.npy"
        np.save(out_path,    mask)
        np.save(gt_out_path, gt_mask)

        metadata.append({
            "idx":          idx,
            "true_label":   label,
            "class_name":   CLASS_NAMES[label],
            "target_class": target,
            "target_name":  CLASS_NAMES[target],
            "mask_path":    str(out_path),
            "gt_mask_path": str(gt_out_path),
        })

        if (idx + 1) % 500 == 0 or (idx + 1) == n:
            print(f"  {idx+1}/{n}", flush=True)

    explainer.remove_hooks()

    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nDone. {n} masks saved to {OUTPUT_DIR}/")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
