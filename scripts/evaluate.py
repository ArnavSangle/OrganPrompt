"""Zero-shot classification evaluation for fine-tuned BioMedCLIP on OrganAMNIST.

Usage:
    # Evaluate fine-tuned checkpoint (default):
    python scripts/evaluate.py

    # Evaluate pretrained baseline (no fine-tuning):
    python scripts/evaluate.py --baseline

    # Custom options:
    python scripts/evaluate.py --checkpoint checkpoints/epoch_009_val3.0132.pt --device cuda --batch_size 128
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

import open_clip
from open_clip import get_tokenizer
from src.data.organmnist import load_organamnist
from src.training.freeze import UNFREEZE_PATTERNS
from src.labels.templates import CLASS_NAMES, ZERO_SHOT_TEMPLATES

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
DEFAULT_CHECKPOINT = "checkpoints/epoch_009_val3.0132.pt"
NUM_CLASSES = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot evaluation of BioMedCLIP on OrganAMNIST")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to fine-tuned checkpoint (relative to project root or absolute)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for image encoding",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Skip checkpoint loading and evaluate the pretrained model with no fine-tuning",
    )
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    """Load BioMedCLIP, optionally applying fine-tuned weights from checkpoint."""
    print(f"Loading BioMedCLIP from Hub...")
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)
    model = model.to(args.device)
    model.eval()

    if args.baseline:
        print("--baseline flag set: using pretrained weights only (no checkpoint loaded).")
        return model

    # Resolve checkpoint path (support relative-to-project-root paths)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(__file__).parent.parent / ckpt_path

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=args.device)
    state_dict = ckpt["model_state_dict"]

    # The checkpoint contains only the unfrozen layers (partial state_dict).
    # strict=False lets us load these without errors for the frozen layers.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # Verify that every missing key is truly a frozen key (expected), and that
    # no checkpoint key was silently ignored.
    frozen_missing = [k for k in missing if not any(pat in k for pat in UNFREEZE_PATTERNS)]
    if frozen_missing:
        print(f"  WARNING: {len(frozen_missing)} frozen keys missing from checkpoint "
              f"(first few: {frozen_missing[:3]})")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys in checkpoint: {unexpected[:3]}")

    loaded = len(state_dict) - len(unexpected)
    print(f"  Loaded {loaded}/{len(state_dict)} checkpoint tensors into model "
          f"(epoch {ckpt.get('epoch', '?')}, val_loss {ckpt.get('val_loss', '?'):.4f})")
    return model


@torch.no_grad()
def encode_text_templates(model: torch.nn.Module, device: str) -> torch.Tensor:
    """Tokenize and encode zero-shot templates for all 11 classes.

    Returns a float32 tensor of shape (11, D) with L2-normalized embeddings.
    """
    tokenizer = get_tokenizer(MODEL_NAME)

    templates = [ZERO_SHOT_TEMPLATES[i] for i in range(NUM_CLASSES)]
    tokens = tokenizer(templates).to(device)   # (11, seq_len)

    text_features = model.encode_text(tokens)  # (11, D)
    text_features = F.normalize(text_features, dim=-1)
    return text_features


@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    test_dataset,
    text_features: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[list[int], list[int]]:
    """Run inference over the full test set.

    Returns (all_labels, all_preds) as plain Python int lists.
    """
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(args.device == "cuda"),
    )

    all_labels: list[int] = []
    all_preds: list[int] = []

    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader):
        # OrganAMNIST returns (image_tensor, label_array, ...)
        images, label_arr = batch[0], batch[1]
        labels = [int(label_arr[i][0]) if hasattr(label_arr[i], '__len__') else int(label_arr[i])
                  for i in range(len(label_arr))]

        images = images.to(args.device)

        image_features = model.encode_image(images)          # (B, D)
        image_features = F.normalize(image_features, dim=-1) # (B, D)

        # Cosine similarity: (B, D) x (D, 11) -> (B, 11)
        logits = image_features @ text_features.T
        preds = logits.argmax(dim=-1).cpu().tolist()

        all_labels.extend(labels)
        all_preds.extend(preds)

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
            print(f"  Batch {batch_idx + 1}/{total_batches} "
                  f"({(batch_idx + 1) * args.batch_size} images processed)...")

    return all_labels, all_preds


def compute_metrics(
    all_labels: list[int],
    all_preds: list[int],
) -> dict:
    """Compute overall accuracy, per-class accuracy, and confusion matrix."""
    n = len(all_labels)
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    overall_acc = correct / n

    # Per-class counts
    class_correct = {i: 0 for i in range(NUM_CLASSES)}
    class_total   = {i: 0 for i in range(NUM_CLASSES)}
    for label, pred in zip(all_labels, all_preds):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1

    per_class_acc = {
        CLASS_NAMES[i]: (class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0)
        for i in range(NUM_CLASSES)
    }

    # Confusion matrix: rows = true class, cols = predicted class
    conf_matrix = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    for label, pred in zip(all_labels, all_preds):
        conf_matrix[label][pred] += 1

    return {
        "overall_accuracy": overall_acc,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": conf_matrix,
        "n_samples": n,
    }


def print_results(metrics: dict, mode: str) -> None:
    """Print a formatted results summary to stdout."""
    print()
    print("=" * 60)
    print(f"ZERO-SHOT EVALUATION RESULTS  [{mode}]")
    print("=" * 60)
    print(f"Total test samples : {metrics['n_samples']}")
    print(f"Overall accuracy   : {metrics['overall_accuracy']:.4f}  "
          f"({metrics['overall_accuracy'] * 100:.2f}%)")
    print()
    print("Per-class accuracy:")
    for class_name, acc in metrics["per_class_accuracy"].items():
        bar = "#" * int(acc * 30)
        print(f"  {class_name:<15s}  {acc:.4f}  {bar}")

    print()
    print("Confusion matrix (rows = true class, cols = predicted class):")
    col_labels = [CLASS_NAMES[i][:6] for i in range(NUM_CLASSES)]
    header = "              " + "  ".join(f"{c:>6}" for c in col_labels)
    print(header)
    print("              " + "-" * (8 * NUM_CLASSES))
    for i, row in enumerate(metrics["confusion_matrix"]):
        row_str = "  ".join(f"{v:>6d}" for v in row)
        print(f"  {CLASS_NAMES[i]:<12s}| {row_str}")
    print("=" * 60)


def main() -> None:
    args = parse_args()

    # Determine evaluation mode label for output files
    mode = "pretrained-baseline" if args.baseline else f"finetuned-{Path(args.checkpoint).stem}"

    print(f"Device     : {args.device}")
    print(f"Batch size : {args.batch_size}")
    print(f"Mode       : {mode}")
    print()

    # Load model
    model = load_model(args)

    # Encode zero-shot text templates
    print("Encoding zero-shot text templates for 11 classes...")
    text_features = encode_text_templates(model, args.device)
    print(f"  Text feature matrix: {text_features.shape}")

    # Load OrganAMNIST test split
    print("Loading OrganAMNIST test split...")
    _, _, test_dataset = load_organamnist(size=224)
    print(f"  Test set size: {len(test_dataset)}")

    # Run inference
    print(f"Running inference in batches of {args.batch_size}...")
    all_labels, all_preds = run_evaluation(model, test_dataset, text_features, args)

    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds)

    # Print results
    print_results(metrics, mode)

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "eval_results.json"

    output = {
        "mode": mode,
        "checkpoint": None if args.baseline else str(args.checkpoint),
        "device": args.device,
        "batch_size": args.batch_size,
        **metrics,
        # Convert class-index keys to class-name keys for JSON readability
        "class_names": CLASS_NAMES,
    }
    # CLASS_NAMES has int keys; convert to str for JSON serialization
    output["class_names"] = {str(k): v for k, v in CLASS_NAMES.items()}

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
