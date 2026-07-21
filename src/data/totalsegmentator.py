import json
import random as _random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ORGAN_FILES = {
    0:  ["urinary_bladder"],
    1:  ["femur_left"],
    2:  ["femur_right"],
    3:  ["heart"],
    4:  ["kidney_left"],
    5:  ["kidney_right"],
    6:  ["liver"],
    7:  ["lung_upper_lobe_left", "lung_lower_lobe_left"],
    8:  ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
    9:  ["pancreas"],
    10: ["spleen"],
}

HU_WINDOW = {
    1:  (-200, 1000),   # femur-left
    2:  (-200, 1000),   # femur-right
    7:  (-1000, -200),  # lung-left
    8:  (-1000, -200),  # lung-right
    "default": (-150, 250),
}


def _build_index(volume_dirs: list[Path], top_k: int) -> list[dict]:
    """Return flat list of {volume_dir, slice_idx, organ_class} for top-k slices per organ."""
    items = []
    for vol_dir in volume_dirs:
        for class_idx, organ_names in ORGAN_FILES.items():
            seg_paths = [vol_dir / "segmentations" / f"{n}.nii.gz" for n in organ_names]
            if not all(p.exists() for p in seg_paths):
                continue
            merged = None
            for p in seg_paths:
                data = np.asarray(nib.load(p).dataobj) > 0
                merged = data if merged is None else (merged | data)
            coverage = merged.sum(axis=(0, 1))  # (n_slices,)
            if coverage.max() == 0:
                continue
            top_indices = np.argsort(coverage, kind="stable")[::-1][:top_k]
            for s in top_indices:
                if coverage[s] > 0:
                    items.append({
                        "volume_dir": str(vol_dir.resolve()),
                        "slice_idx":  int(s),
                        "organ_class": int(class_idx),
                    })
    return items


class TotalSegmentatorDataset(Dataset):
    """Axial CT slices from TotalSegmentator volumes.

    Each item in training mode: (image_tensor (3,224,224), label_int)
    Each item in eval mode:     (image_tensor (3,224,224), label_int, gt_mask (224,224))
    """

    def __init__(self, items: list[dict], eval_mode: bool = False, include_masks: bool = False):
        self.items = items
        self.eval_mode = eval_mode
        self._return_masks = eval_mode or include_masks

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        organ_class = item["organ_class"]

        if "slice_path" in item:
            # Fast path: precomputed .npy slice (float16 on disk → float32 in memory)
            arr = np.load(item["slice_path"]).astype(np.float32)  # (224, 224)
            img = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)
        else:
            # Slow path: decompress full NIfTI volume (use precompute_slices.py to avoid this)
            vol_dir = Path(item["volume_dir"])
            slice_idx = item["slice_idx"]
            ct_vol = np.asarray(nib.load(vol_dir / "ct.nii.gz").dataobj)
            ct_slice = ct_vol[:, :, slice_idx].astype(np.float32)
            hu_min, hu_max = HU_WINDOW.get(organ_class, HU_WINDOW["default"])
            ct_slice = np.clip(ct_slice, hu_min, hu_max)
            ct_slice = (ct_slice - hu_min) / (hu_max - hu_min)
            img = torch.from_numpy(ct_slice).unsqueeze(0).repeat(3, 1, 1)
            img = F.interpolate(
                img.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
            ).squeeze(0)

        if self._return_masks:
            vol_dir = Path(item["volume_dir"])
            gt = self._load_gt_mask(vol_dir, organ_class, item["slice_idx"])
            return img, organ_class, gt

        return img, organ_class

    def _load_gt_mask(self, vol_dir: Path, organ_class: int, slice_idx: int) -> np.ndarray:
        organ_names = ORGAN_FILES[organ_class]
        merged = None
        for name in organ_names:
            seg = np.asarray(nib.load(vol_dir / "segmentations" / f"{name}.nii.gz").dataobj)
            sl = (seg[:, :, slice_idx] > 0)
            merged = sl if merged is None else (merged | sl)
        gt = merged.astype(np.float32)
        gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
        gt_resized = F.interpolate(gt_t, size=(224, 224), mode="nearest").squeeze().numpy()
        return gt_resized.astype(np.float32)


def load_totalsegmentator(
    root: str | Path,
    top_k: int = 5,
    seed: int = 42,
    cache_path: str | Path | None = None,
    include_masks: bool = False,
) -> tuple[Dataset, Dataset, Dataset]:
    """Return (train_ds, val_ds, test_ds) from TotalSegmentator.

    root: path to Totalsegmentator_dataset_v201/
    top_k: slices per organ per volume (default 5)
    seed: RNG seed for volume-level 80/10/10 split (default 42)
    cache_path: where to save/load the slice index JSON (default: root/totalseg_slice_index.json)
    """
    root = Path(root).resolve()
    if cache_path is None:
        cache_path = root / "totalseg_slice_index.json"
    cache_path = Path(cache_path)

    volume_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("s")])
    rng = _random.Random(seed)
    dirs = list(volume_dirs)
    rng.shuffle(dirs)
    n = len(dirs)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    split_dirs = {
        "train": dirs[:n_train],
        "val":   dirs[n_train:n_train + n_val],
        "test":  dirs[n_train + n_val:],
    }

    if cache_path.exists():
        with open(cache_path) as f:
            all_items = json.load(f)
        vol_to_split = {
            str(d.resolve()): name
            for name, dlist in split_dirs.items()
            for d in dlist
        }
        split_items: dict[str, list] = {"train": [], "val": [], "test": []}
        for item in all_items:
            sp = vol_to_split.get(item["volume_dir"])
            if sp:
                split_items[sp].append(item)
    else:
        print(f"Building slice index for {n} volumes (first run — may take several minutes)...")
        split_items = {}
        for split_name, sdirs in split_dirs.items():
            print(f"  Indexing {split_name} ({len(sdirs)} volumes)...")
            split_items[split_name] = _build_index(sdirs, top_k)
        all_items = [item for items in split_items.values() for item in items]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(all_items, f)
        print(f"Saved slice index to {cache_path}")

    return (
        TotalSegmentatorDataset(split_items["train"], include_masks=include_masks),
        TotalSegmentatorDataset(split_items["val"],   include_masks=include_masks),
        TotalSegmentatorDataset(split_items["test"],  eval_mode=True),
    )
