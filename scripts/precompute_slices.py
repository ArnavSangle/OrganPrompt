"""Pre-extract CT slices from TotalSegmentator NIfTI volumes to .npy files.

Run once before training. Groups items by volume so each CT is decompressed
only once. Updates totalseg_slice_index.json with slice_path fields so
TotalSegmentatorDataset.__getitem__ uses fast .npy loading instead of
decompressing the full .nii.gz on every sample.

Usage:
    python scripts/precompute_slices.py --root Totalsegmentator_dataset_v201
    python scripts/precompute_slices.py --root Totalsegmentator_dataset_v201 --out_dir my_slices
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.totalsegmentator import HU_WINDOW, load_totalsegmentator


def _apply_window_resize(ct_vol: np.ndarray, slice_idx: int, organ_class: int) -> np.ndarray:
    ct_slice = ct_vol[:, :, slice_idx].astype(np.float32)
    hu_min, hu_max = HU_WINDOW.get(organ_class, HU_WINDOW["default"])
    ct_slice = np.clip(ct_slice, hu_min, hu_max)
    ct_slice = (ct_slice - hu_min) / (hu_max - hu_min)
    t = torch.from_numpy(ct_slice).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    return t.squeeze().numpy().astype(np.float16)  # float16 halves disk usage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                        help="Path to Totalsegmentator_dataset_v201/")
    parser.add_argument("--out_dir", default=None,
                        help="Where to save .npy slices (default: root/precomputed/)")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root / "precomputed"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = root / "totalseg_slice_index.json"
    if not cache_path.exists():
        print("Slice index not found — building it first (this takes a few minutes)...")
        load_totalsegmentator(root, top_k=args.top_k)

    with open(cache_path) as f:
        all_items = json.load(f)

    already_done = sum(1 for item in all_items if "slice_path" in item)
    if already_done == len(all_items):
        print(f"All {len(all_items)} slices already precomputed. Nothing to do.")
        return
    if already_done > 0:
        print(f"Resuming: {already_done}/{len(all_items)} already done.")

    # Group items by volume so we only decompress each .nii.gz once
    by_volume: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(all_items):
        if "slice_path" not in item:
            by_volume[item["volume_dir"]].append(i)

    n_volumes = len(by_volume)
    print(f"Processing {len(all_items) - already_done} slices across {n_volumes} volumes...")
    print(f"Output dir: {out_dir}")
    print(f"Estimated disk usage: ~{(len(all_items) * 224 * 224 * 2) / 1e9:.1f} GB (float16)\n")

    for vol_i, (vol_dir_str, item_indices) in enumerate(by_volume.items()):
        vol_dir = Path(vol_dir_str)
        ct_path = vol_dir / "ct.nii.gz"

        print(f"[{vol_i+1}/{n_volumes}] {vol_dir.name} ({len(item_indices)} slices)...", end=" ", flush=True)
        ct_vol = np.asarray(nib.load(ct_path).dataobj)

        for i in item_indices:
            item = all_items[i]
            fname = f"{vol_dir.name}_c{item['organ_class']}_s{item['slice_idx']}.npy"
            slice_path = out_dir / fname
            if not slice_path.exists():
                arr = _apply_window_resize(ct_vol, item["slice_idx"], item["organ_class"])
                np.save(slice_path, arr)
            all_items[i]["slice_path"] = str(slice_path)

        print("done")

    with open(cache_path, "w") as f:
        json.dump(all_items, f)

    total_gb = sum(
        Path(item["slice_path"]).stat().st_size
        for item in all_items if "slice_path" in item
    ) / 1e9
    print(f"\nDone. {len(all_items)} slices saved ({total_gb:.1f} GB).")
    print(f"Index updated: {cache_path}")
    print("\nRun training now — it will use precomputed slices automatically.")


if __name__ == "__main__":
    main()
