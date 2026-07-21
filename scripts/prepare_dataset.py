"""Prepare OrganAMNIST image-text pairs for contrastive training.

Outputs data/prepared/captions.json with structure:
{
  "class_names": {"0": "bladder", ...},
  "splits": {
    "train": [{"idx": 0, "label": 6, "class_name": "liver", "zero_shot": "...", "few_shot": "...", "variants": ["...", "...", "..."]}, ...],
    "val":   [...],
    "test":  [...]
  }
}

Runs Claude API to generate 3 variants per class (not per image).
If ANTHROPIC_API_KEY is not set, variants fall back to the zero_shot template repeated 3 times.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.organmnist import load_organamnist
from src.labels.templates import ZERO_SHOT_TEMPLATES, FEW_SHOT_TEMPLATES, CLASS_NAMES
from src.labels.generator import generate_caption_variants

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "prepared" / "captions.json"
N_VARIANTS = 3


def build_class_captions() -> dict[int, dict]:
    print("Generating captions (one API call per class if key is set)...")
    class_captions = {}
    for class_idx in range(11):
        name = CLASS_NAMES[class_idx]
        zero = ZERO_SHOT_TEMPLATES[class_idx]
        few  = FEW_SHOT_TEMPLATES[class_idx]

        if os.getenv("ANTHROPIC_API_KEY"):
            variants = generate_caption_variants(zero, n=N_VARIANTS)
        else:
            print(f"  [WARN] No API key — using zero_shot as variants for class {class_idx} ({name})")
            variants = [zero] * N_VARIANTS

        class_captions[class_idx] = {
            "name": name,
            "zero_shot": zero,
            "few_shot": few,
            "variants": variants,
        }
        print(f"  [{class_idx:2d}] {name}: done")
    return class_captions


def build_split_records(dataset, split_name: str, class_captions: dict) -> list[dict]:
    print(f"Building records for {split_name} split ({len(dataset)} images)...")
    records = []
    for idx in range(len(dataset)):
        _, label_arr = dataset[idx]
        label = int(label_arr[0]) if hasattr(label_arr, '__len__') else int(label_arr)
        cap = class_captions[label]
        records.append({
            "idx": idx,
            "label": label,
            "class_name": cap["name"],
            "zero_shot": cap["zero_shot"],
            "few_shot": cap["few_shot"],
            "variants": cap["variants"],
        })
    return records


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    class_captions = build_class_captions()
    train, val, test = load_organamnist(size=224)

    output = {
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "splits": {
            "train": build_split_records(train, "train", class_captions),
            "val":   build_split_records(val,   "val",   class_captions),
            "test":  build_split_records(test,  "test",  class_captions),
        }
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    total = sum(len(v) for v in output["splits"].values())
    print(f"\nSaved {total} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
