import pytest
import numpy as np
import nibabel as nib
import torch
from pathlib import Path

from src.data.totalsegmentator import (
    ORGAN_FILES, HU_WINDOW, _build_index,
    TotalSegmentatorDataset, load_totalsegmentator,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from evaluate_masks import _compute_iou

ALL_SEG_NAMES = [
    "urinary_bladder", "femur_left", "femur_right", "heart",
    "kidney_left", "kidney_right", "liver",
    "lung_upper_lobe_left", "lung_lower_lobe_left",
    "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
    "pancreas", "spleen",
]


@pytest.fixture
def fake_root(tmp_path):
    """Minimal fake TotalSegmentator tree: 2 volumes, liver present only."""
    affine = np.eye(4)
    shape = (64, 64, 20)
    for vol_name in ["s0001", "s0002"]:
        vol_dir = tmp_path / vol_name
        seg_dir = vol_dir / "segmentations"
        seg_dir.mkdir(parents=True)
        ct = np.random.uniform(-500, 500, shape).astype(np.float32)
        nib.save(nib.Nifti1Image(ct, affine), str(vol_dir / "ct.nii.gz"))
        for name in ALL_SEG_NAMES:
            mask = np.zeros(shape, dtype=np.uint8)
            if name == "liver":
                mask[20:40, 20:40, 5:15] = 1  # organ in slices 5-14
            nib.save(nib.Nifti1Image(mask, affine), str(seg_dir / f"{name}.nii.gz"))
    return tmp_path


def test_organ_files_covers_all_classes():
    assert set(ORGAN_FILES.keys()) == set(range(11))


def test_hu_window_has_default_and_special_cases():
    assert "default" in HU_WINDOW
    assert HU_WINDOW.get(7) == (-1000, -200)   # lung-left
    assert HU_WINDOW.get(8) == (-1000, -200)   # lung-right
    assert HU_WINDOW.get(1) == (-200, 1000)    # femur-left
    assert HU_WINDOW.get(2) == (-200, 1000)    # femur-right


def test_build_index_returns_liver_slices(fake_root):
    vol_dir = fake_root / "s0001"
    items = _build_index([vol_dir], top_k=5)
    liver_items = [i for i in items if i["organ_class"] == 6]
    assert len(liver_items) == 5
    assert all(5 <= i["slice_idx"] < 15 for i in liver_items)


def test_build_index_skips_absent_organs(fake_root):
    vol_dir = fake_root / "s0001"
    items = _build_index([vol_dir], top_k=5)
    # only liver has non-zero mask → only liver items
    assert all(i["organ_class"] == 6 for i in items)


def test_dataset_getitem_shape(fake_root):
    vol_dir = fake_root / "s0001"
    items = _build_index([vol_dir], top_k=2)
    assert len(items) >= 1
    ds = TotalSegmentatorDataset(items[:1])
    img, label = ds[0]
    assert img.shape == (3, 224, 224)
    assert img.dtype == torch.float32
    assert 0 <= label <= 10


def test_dataset_image_in_zero_one(fake_root):
    vol_dir = fake_root / "s0001"
    items = _build_index([vol_dir], top_k=2)
    ds = TotalSegmentatorDataset(items[:1])
    img, _ = ds[0]
    assert float(img.min()) >= 0.0
    assert float(img.max()) <= 1.0


def test_dataset_eval_mode_gt_mask(fake_root):
    vol_dir = fake_root / "s0001"
    items = _build_index([vol_dir], top_k=2)
    ds = TotalSegmentatorDataset(items[:1], eval_mode=True)
    img, label, gt = ds[0]
    assert img.shape == (3, 224, 224)
    assert gt.shape == (224, 224)
    assert gt.dtype == np.float32
    assert set(np.unique(gt)).issubset({0.0, 1.0})


def test_load_totalsegmentator_splits(fake_root):
    train_ds, val_ds, test_ds = load_totalsegmentator(fake_root, top_k=3, seed=42)
    total = len(train_ds) + len(val_ds) + len(test_ds)
    assert total > 0
    # No volume appears in two splits: check volume_dirs don't overlap
    def vol_dirs(ds):
        return {item["volume_dir"] for item in ds.items}
    assert vol_dirs(train_ds).isdisjoint(vol_dirs(val_ds))
    assert vol_dirs(train_ds).isdisjoint(vol_dirs(test_ds))
    assert vol_dirs(val_ds).isdisjoint(vol_dirs(test_ds))


def test_iou_perfect():
    """IoU of identical binary masks should be 1.0."""
    mask = np.ones((224, 224), dtype=np.float32)
    iou = _compute_iou(mask > 0.5, mask > 0.5)
    assert abs(iou - 1.0) < 1e-6


def test_iou_no_overlap():
    """IoU of non-overlapping masks should be 0.0."""
    pred = np.zeros((224, 224), dtype=bool)
    pred[:112, :] = True
    gt = np.zeros((224, 224), dtype=bool)
    gt[112:, :] = True
    iou = _compute_iou(pred, gt)
    assert abs(iou) < 1e-6


def test_iou_empty_gt_returns_none():
    """IoU when GT is empty should return None (skip)."""
    pred = np.ones((224, 224), dtype=bool)
    gt   = np.zeros((224, 224), dtype=bool)
    result = _compute_iou(pred, gt)
    assert result is None
