# Bone-Metastasis–Organ Index (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 3D bone-metastasis–to-organ proximity/overlap index on Spine-Mets-CT-SEG, using ground-truth lesion masks for tumors and the existing OrganCAM v5 model for organ maps.

**Architecture:** A DICOM loader turns each Spine-Mets patient into (CT HU volume, spacing, tumor mask, per-vertebra lesion type). The OrganCAM v5 patch-similarity model runs per axial slice and the 2D organ maps are stacked into a 3D organ-probability volume aligned to the CT grid. A pure-geometry index module computes per-(lesion, organ) overlap and surface-distance proximity, blends them, and aggregates to a per-patient score. An eval script validates the organ maps against CT-ORG organ GT.

**Tech Stack:** Python 3.13, PyTorch, open_clip (BioMedCLIP), pydicom + pydicom-seg (DICOM/SEG), openpyxl (clinical xlsx), numpy, scipy.ndimage (distance transforms), nibabel (CT-ORG NIfTI), pytest.

## Global Constraints

- Reuse OrganCAM primitives unchanged: `BioMedCLIPPatchSimilarity` (`src/explainability/patch_similarity.py`), `CLASS_NAMES` + `FEW_SHOT_TEMPLATES` (`src/labels/templates.py`), `HU_WINDOW` (`src/data/totalsegmentator.py`), `UNFREEZE_PATTERNS` (`src/training/freeze.py`).
- Model id: `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`. Checkpoint: `checkpoints/totalseg_patch_v5/epoch_011_val3.4584.pt`.
- OrganCAM classes are 11 organs, indices 0–10 per `CLASS_NAMES`.
- Slices are processed at 224×224; organ probability = raw patch-text cosine similarity (no per-image normalization), matching current inference.
- All 3D geometry uses physical voxel spacing (mm); never mix voxel and mm units.
- New code lives under `src/tumor/` and `src/data/spine_mets.py`; new scripts under `scripts/`; tests under `tests/`. Do not restructure existing OrganCAM files.
- Commit after every task. Do not push (user pushes manually). No `Co-Authored-By`/AI attribution in commit messages.
- Test data root: `data/Spine-Mets-CT-SEG_v1_2024/`. Clinical labels: `data/Spine-Mets-CT-SEG_v1_2024/Spine-Mets-CT-SEG_Clinical.xlsx` (already downloaded).

---

### Task 1: Dependencies + package skeleton

**Files:**
- Create: `src/tumor/__init__.py`
- Modify: `README.md` (Setup section — append install line)

**Interfaces:**
- Produces: importable `src.tumor` package; `pydicom`, `pydicom_seg`, `openpyxl` available.

- [ ] **Step 1: Install runtime deps**

Run: `.venv/Scripts/python.exe -m pip install pydicom pydicom-seg openpyxl`
Expected: `Successfully installed ...` (pydicom and openpyxl may already be present).

- [ ] **Step 2: Verify imports**

Run: `.venv/Scripts/python.exe -c "import pydicom, pydicom_seg, openpyxl, scipy.ndimage, nibabel; print('deps ok')"`
Expected: `deps ok`

- [ ] **Step 3: Create the package init**

```python
# src/tumor/__init__.py
"""Bone-metastasis to organ proximity/overlap index (Phase 1)."""
```

- [ ] **Step 4: Append dependency note to README Setup**

Add under the existing `pip install ...` line in `README.md`:

```
pip install pydicom pydicom-seg openpyxl   # bone-metastasis index (Phase 1)
```

- [ ] **Step 5: Commit**

```bash
git add src/tumor/__init__.py README.md
git commit -m "feat: scaffold src/tumor package and add bone-index deps"
```

---

### Task 2: Parse the clinical lesion-label spreadsheet

**Files:**
- Create: `src/data/spine_mets_labels.py`
- Test: `tests/test_spine_mets_labels.py`

**Interfaces:**
- Produces:
  - `VERTEBRA_LEVELS: list[str]` — `["C1"..."C7","T1"..."T12","L1"..."L5","S1"]`.
  - `parse_levels(cell: str | None) -> list[str]` — split a spreadsheet cell like `"T1, T2 T3"` into `["T1","T2","T3"]`, uppercased, deduped, only valid levels.
  - `load_lesion_labels(xlsx_path: str) -> dict[str, dict[str, str]]` — returns `{case_id: {level: lesion_type}}` where `lesion_type ∈ {"lytic","blastic","mixed"}`; `case_id` is the string in the `Case` column.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_spine_mets_labels.py
from pathlib import Path
import pytest
from src.data.spine_mets_labels import parse_levels, load_lesion_labels, VERTEBRA_LEVELS

XLSX = Path("data/Spine-Mets-CT-SEG_v1_2024/Spine-Mets-CT-SEG_Clinical.xlsx")

def test_parse_levels_splits_commas_and_spaces():
    assert parse_levels("T1, T2 T3, T4") == ["T1", "T2", "T3", "T4"]

def test_parse_levels_handles_empty():
    assert parse_levels(None) == []
    assert parse_levels("") == []
    assert parse_levels("No lesions observed") == []  # no valid level tokens

def test_parse_levels_dedupes_and_uppercases():
    assert parse_levels("t1, T1, l5") == ["T1", "L5"]

def test_vertebra_levels_cover_thoracolumbar():
    for lvl in ["T1", "T12", "L1", "L5"]:
        assert lvl in VERTEBRA_LEVELS

@pytest.mark.skipif(not XLSX.exists(), reason="clinical xlsx not present")
def test_load_lesion_labels_real_file():
    labels = load_lesion_labels(str(XLSX))
    assert "10355" in labels                       # known Mixed-lesion patient
    assert any(t == "mixed" for t in labels["10355"].values())
    # patient 10352 recorded "No lesions observed"
    assert labels.get("10352", {}) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_spine_mets_labels.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.spine_mets_labels'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/spine_mets_labels.py
import re
import openpyxl

VERTEBRA_LEVELS = (
    [f"C{i}" for i in range(1, 8)]
    + [f"T{i}" for i in range(1, 13)]
    + [f"L{i}" for i in range(1, 6)]
    + ["S1"]
)
_LEVEL_RE = re.compile(r"\b([CTLS])\s*0*([0-9]{1,2})\b", re.IGNORECASE)
_VALID = set(VERTEBRA_LEVELS)

def parse_levels(cell):
    if not cell:
        return []
    out = []
    for prefix, num in _LEVEL_RE.findall(str(cell)):
        lvl = f"{prefix.upper()}{int(num)}"
        if lvl in _VALID and lvl not in out:
            out.append(lvl)
    return out

def load_lesion_labels(xlsx_path):
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb["Spine-Mets-CT-SEG"]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(c).strip() if c is not None else "" for c in rows[0]]
    idx = {name: header.index(name) for name in ("Case", "Blastic", "Lytic", "Mixed")}
    labels = {}
    for r in rows[1:]:
        case = r[idx["Case"]]
        if case is None:
            continue
        case = str(case).strip()
        per_level = {}
        for kind in ("blastic", "lytic", "mixed"):
            for lvl in parse_levels(r[idx[kind.capitalize()]]):
                per_level[lvl] = kind      # last-writer wins; mixed listed last
        labels[case] = per_level
    return labels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_spine_mets_labels.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/data/spine_mets_labels.py tests/test_spine_mets_labels.py
git commit -m "feat: parse Spine-Mets clinical lesion-level labels"
```

---

### Task 3: Locate CT and SEG series within a patient folder

**Files:**
- Create: `src/data/spine_mets.py`
- Test: `tests/test_spine_mets_series.py`

**Interfaces:**
- Consumes: patient dir under `data/Spine-Mets-CT-SEG_v1_2024/spine_mets_ct_seg/<case>/`.
- Produces:
  - `find_series(patient_dir: str) -> tuple[list[str], str]` — returns `(ct_dicom_paths_sorted_by_z, seg_dicom_path)`. CT = the series directory with the most `.dcm` files; SEG = the single file whose `Modality == "SEG"`. CT paths sorted ascending by `ImagePositionPatient[2]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_spine_mets_series.py
from pathlib import Path
import pytest
from src.data.spine_mets import find_series

ROOT = Path("data/Spine-Mets-CT-SEG_v1_2024/spine_mets_ct_seg")

@pytest.mark.skipif(not ROOT.exists(), reason="Spine-Mets not present")
def test_find_series_returns_ct_and_seg():
    case = sorted(p for p in ROOT.iterdir() if p.is_dir())[0]
    ct, seg = find_series(str(case))
    assert len(ct) > 100                       # a full CT stack
    assert seg.endswith(".dcm")
    import pydicom
    assert pydicom.dcmread(seg, stop_before_pixels=True).Modality == "SEG"
    # CT sorted ascending by z
    zs = [float(pydicom.dcmread(p, stop_before_pixels=True).ImagePositionPatient[2]) for p in ct[:5]]
    assert zs == sorted(zs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_spine_mets_series.py -q`
Expected: FAIL with `ImportError: cannot import name 'find_series'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/spine_mets.py
import glob
import os
from collections import Counter
import pydicom

def find_series(patient_dir):
    dcms = glob.glob(os.path.join(patient_dir, "**", "*.dcm"), recursive=True)
    seg_path = None
    ct_candidates = []
    for p in dcms:
        d = pydicom.dcmread(p, stop_before_pixels=True)
        if getattr(d, "Modality", None) == "SEG":
            seg_path = p
        elif getattr(d, "Modality", None) == "CT":
            ct_candidates.append(p)
    if seg_path is None or not ct_candidates:
        raise FileNotFoundError(f"CT or SEG series missing in {patient_dir}")
    # CT series = directory holding the most CT slices
    by_dir = Counter(os.path.dirname(p) for p in ct_candidates)
    ct_dir = by_dir.most_common(1)[0][0]
    ct = [p for p in ct_candidates if os.path.dirname(p) == ct_dir]
    ct.sort(key=lambda p: float(pydicom.dcmread(p, stop_before_pixels=True).ImagePositionPatient[2]))
    return ct, seg_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_spine_mets_series.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/spine_mets.py tests/test_spine_mets_series.py
git commit -m "feat: locate CT and SEG series in a Spine-Mets patient folder"
```

---

### Task 4: Load CT volume (HU + spacing) and per-vertebra SEG masks

**Files:**
- Modify: `src/data/spine_mets.py`
- Test: `tests/test_spine_mets_volume.py`

**Interfaces:**
- Consumes: `find_series` (Task 3).
- Produces (added to `src/data/spine_mets.py`):
  - `load_ct_volume(ct_paths: list[str]) -> tuple[np.ndarray, tuple[float,float,float]]` — returns `(hu_volume[D,H,W] float32, spacing_zyx_mm)`. HU = `pixel_array*RescaleSlope + RescaleIntercept`. `spacing_zyx = (slice_gap, PixelSpacing[0], PixelSpacing[1])` where `slice_gap` = median abs diff of consecutive `ImagePositionPatient[2]`.
  - `load_vertebra_masks(seg_path: str, ct_paths: list[str]) -> dict[str, np.ndarray]` — `{level: bool_mask[D,H,W]}` resampled onto the CT grid via `pydicom_seg.SegmentReader` and per-frame referenced-instance mapping; segment level from `SegmentLabel` (e.g. `"T1 vertebra" -> "T1"`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_spine_mets_volume.py
from pathlib import Path
import numpy as np
import pytest
from src.data.spine_mets import find_series, load_ct_volume, load_vertebra_masks

ROOT = Path("data/Spine-Mets-CT-SEG_v1_2024/spine_mets_ct_seg")

@pytest.mark.skipif(not ROOT.exists(), reason="Spine-Mets not present")
def test_ct_volume_and_masks_align():
    case = sorted(p for p in ROOT.iterdir() if p.is_dir())[0]
    ct, seg = find_series(str(case))
    vol, spacing = load_ct_volume(ct)
    assert vol.ndim == 3 and vol.shape[1:] == (512, 512)
    assert vol.min() < -500 and vol.max() > 500      # real HU range
    assert all(s > 0 for s in spacing) and len(spacing) == 3
    masks = load_vertebra_masks(seg, ct)
    assert len(masks) >= 5                            # several vertebrae
    any_level = next(iter(masks))
    assert masks[any_level].shape == vol.shape        # aligned to CT grid
    assert masks[any_level].dtype == bool
    assert masks[any_level].sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_spine_mets_volume.py -q`
Expected: FAIL with `ImportError: cannot import name 'load_ct_volume'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/data/spine_mets.py
import numpy as np
import pydicom_seg

def load_ct_volume(ct_paths):
    slices = [pydicom.dcmread(p) for p in ct_paths]
    slope = float(getattr(slices[0], "RescaleSlope", 1))
    inter = float(getattr(slices[0], "RescaleIntercept", 0))
    vol = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=0)
    vol = vol * slope + inter
    zs = [float(s.ImagePositionPatient[2]) for s in slices]
    dz = float(np.median(np.abs(np.diff(zs)))) if len(zs) > 1 else 1.0
    ps = [float(x) for x in slices[0].PixelSpacing]
    return vol.astype(np.float32), (dz, ps[0], ps[1])

def _level_from_label(label):
    # "T1 vertebra" -> "T1"; returns None if not a vertebra segment
    tok = str(label).split()[0].upper()
    return tok if tok and tok[0] in "CTLS" and tok[1:].isdigit() else None

def load_vertebra_masks(seg_path, ct_paths):
    seg = pydicom.dcmread(seg_path)
    reader = pydicom_seg.SegmentReader()
    result = reader.read(seg)
    # Map each CT slice's SOPInstanceUID to its index in the sorted CT stack
    sop_to_z = {}
    for i, p in enumerate(ct_paths):
        d = pydicom.dcmread(p, stop_before_pixels=True)
        sop_to_z[d.SOPInstanceUID] = i
    D = len(ct_paths)
    H = int(seg.Rows)
    W = int(seg.Columns)
    masks = {}
    for seg_num in result.available_segments:
        label = seg.SegmentSequence[seg_num - 1].SegmentLabel
        level = _level_from_label(label)
        if level is None:
            continue
        seg_vol = result.segment_data(seg_num)          # (frames, H, W) in SEG frame order
        # pydicom_seg exposes per-frame referenced SOP via result; fall back to
        # PerFrameFunctionalGroupsSequence for CT-grid placement.
        out = np.zeros((D, H, W), dtype=bool)
        frames = seg.PerFrameFunctionalGroupsSequence
        for f_idx, fg in enumerate(frames):
            ref = fg.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
            z = sop_to_z.get(ref)
            seg_of_frame = fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber
            if z is not None and seg_of_frame == seg_num:
                out[z] |= seg.pixel_array[f_idx].astype(bool)
        if out.any():
            masks[level] = out
    return masks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_spine_mets_volume.py -q`
Expected: PASS

Note: if `pydicom_seg`'s API differs, the per-frame `PerFrameFunctionalGroupsSequence` placement (the loop above) is authoritative and independent of it — it reads `seg.pixel_array` frames directly.

- [ ] **Step 5: Commit**

```bash
git add src/data/spine_mets.py tests/test_spine_mets_volume.py
git commit -m "feat: load Spine-Mets CT volume and per-vertebra SEG masks on the CT grid"
```

---

### Task 5: Assemble a patient into (CT, tumor mask, spacing)

**Files:**
- Modify: `src/data/spine_mets.py`
- Test: `tests/test_spine_mets_patient.py`

**Interfaces:**
- Consumes: `find_series`, `load_ct_volume`, `load_vertebra_masks` (Tasks 3–4), `load_lesion_labels` (Task 2).
- Produces:
  - `@dataclass SpineMetsPatient` with fields `case: str`, `ct: np.ndarray[D,H,W]`, `spacing: tuple[float,float,float]`, `tumor_mask: np.ndarray[D,H,W] bool`, `lesion_type: np.ndarray[D,H,W] int` (0=none,1=lytic,2=blastic,3=mixed).
  - `load_patient(root: str, case: str, labels: dict) -> SpineMetsPatient` — tumor_mask = union of vertebra masks for levels present in `labels[case]`; `lesion_type` encodes each voxel's type.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_spine_mets_patient.py
from pathlib import Path
import numpy as np
import pytest
from src.data.spine_mets import load_patient
from src.data.spine_mets_labels import load_lesion_labels

ROOT = Path("data/Spine-Mets-CT-SEG_v1_2024")
CASES = ROOT / "spine_mets_ct_seg"

@pytest.mark.skipif(not (ROOT / "Spine-Mets-CT-SEG_Clinical.xlsx").exists(), reason="data absent")
def test_load_patient_builds_tumor_mask():
    labels = load_lesion_labels(str(ROOT / "Spine-Mets-CT-SEG_Clinical.xlsx"))
    # pick a case that (a) exists on disk and (b) has >=1 labeled level
    case = next(c.name for c in sorted(CASES.iterdir())
                if c.is_dir() and labels.get(c.name))
    p = load_patient(str(ROOT), case, labels)
    assert p.ct.shape == p.tumor_mask.shape == p.lesion_type.shape
    assert p.tumor_mask.sum() > 0
    assert set(np.unique(p.lesion_type)).issubset({0, 1, 2, 3})
    assert p.lesion_type[p.tumor_mask].min() >= 1     # labeled where tumor is
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_spine_mets_patient.py -q`
Expected: FAIL with `ImportError: cannot import name 'load_patient'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/data/spine_mets.py
import os
from dataclasses import dataclass

_TYPE_CODE = {"lytic": 1, "blastic": 2, "mixed": 3}

@dataclass
class SpineMetsPatient:
    case: str
    ct: np.ndarray
    spacing: tuple
    tumor_mask: np.ndarray
    lesion_type: np.ndarray

def load_patient(root, case, labels):
    patient_dir = os.path.join(root, "spine_mets_ct_seg", case)
    ct_paths, seg_path = find_series(patient_dir)
    ct, spacing = load_ct_volume(ct_paths)
    vmasks = load_vertebra_masks(seg_path, ct_paths)
    tumor = np.zeros(ct.shape, dtype=bool)
    ltype = np.zeros(ct.shape, dtype=np.int8)
    for level, kind in labels.get(case, {}).items():
        m = vmasks.get(level)
        if m is None:
            continue
        tumor |= m
        ltype[m] = _TYPE_CODE[kind]
    return SpineMetsPatient(case, ct, spacing, tumor, ltype)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_spine_mets_patient.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/spine_mets.py tests/test_spine_mets_patient.py
git commit -m "feat: assemble Spine-Mets patient into CT + tumor mask + lesion-type volume"
```

---

### Task 6: Slice preprocessing for OrganCAM input

**Files:**
- Create: `src/tumor/organ_volume.py`
- Test: `tests/test_slice_preprocess.py`

**Interfaces:**
- Produces:
  - `preprocess_slice(hu_slice: np.ndarray, organ_class: int) -> torch.Tensor` — replicate OrganCAM's slow-path preprocessing: HU-window per `HU_WINDOW[organ_class]` (default `(-150,250)`), normalize to `[0,1]`, resize to 224×224 bilinear, repeat to 3 channels. Returns `(3,224,224)` float32.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_slice_preprocess.py
import numpy as np
import torch
from src.tumor.organ_volume import preprocess_slice

def test_preprocess_slice_shape_and_range():
    hu = np.random.uniform(-1000, 1000, size=(512, 512)).astype(np.float32)
    t = preprocess_slice(hu, organ_class=6)      # liver -> default window
    assert isinstance(t, torch.Tensor)
    assert t.shape == (3, 224, 224)
    assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0

def test_preprocess_slice_lung_window_differs():
    hu = np.full((512, 512), -600.0, np.float32)   # lung-ish HU
    liver = preprocess_slice(hu, 6)                # window (-150,250) -> clipped low
    lung = preprocess_slice(hu, 7)                 # window (-1000,-200) -> mid-range
    assert float(lung.mean()) > float(liver.mean())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_slice_preprocess.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tumor.organ_volume'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/tumor/organ_volume.py
import numpy as np
import torch
import torch.nn.functional as F
from src.data.totalsegmentator import HU_WINDOW

def preprocess_slice(hu_slice, organ_class):
    lo, hi = HU_WINDOW.get(organ_class, HU_WINDOW["default"])
    x = np.clip(hu_slice.astype(np.float32), lo, hi)
    x = (x - lo) / (hi - lo)
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)          # (1,1,H,W)
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    return t.squeeze(0).repeat(3, 1, 1)                        # (3,224,224)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_slice_preprocess.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tumor/organ_volume.py tests/test_slice_preprocess.py
git commit -m "feat: OrganCAM-compatible CT slice preprocessing"
```

---

### Task 7: Build the 3D organ-probability volume from OrganCAM

**Files:**
- Modify: `src/tumor/organ_volume.py`
- Test: `tests/test_organ_volume.py`

**Interfaces:**
- Consumes: `preprocess_slice` (Task 6); `BioMedCLIPPatchSimilarity`, `CLASS_NAMES`, `FEW_SHOT_TEMPLATES`.
- Produces:
  - `load_organcam(checkpoint: str, device: str) -> tuple[model, text_features]` — load base BioMedCLIP, apply checkpoint (`strict=False`), encode the 11 `FEW_SHOT_TEMPLATES` → `(11,512)` normalized text features.
  - `build_organ_volume(ct: np.ndarray, model, text_features, device, organs: list[int] | None = None) -> np.ndarray` — for each slice and each organ, run `BioMedCLIPPatchSimilarity.generate` at 224 and downsample back to the CT slice `(H,W)`; return `float32[len(organs), D, H, W]` of raw similarities. Default organs = all 11.

- [ ] **Step 1: Write the failing test** (uses a tiny fake model — no GPU/network)

```python
# tests/test_organ_volume.py
import numpy as np
import torch
from src.tumor import organ_volume as ov

class _FakeExplainer:
    def __init__(self, model, device="cpu"): pass
    def generate(self, image, text_features, target_class, image_size=224):
        # deterministic: constant map equal to class index / 10
        return np.full((image_size, image_size), target_class / 10.0, np.float32)
    def remove_hooks(self): pass

def test_build_organ_volume_shape_and_downsample(monkeypatch):
    monkeypatch.setattr(ov, "BioMedCLIPPatchSimilarity", _FakeExplainer)
    ct = np.random.uniform(-500, 500, size=(4, 512, 512)).astype(np.float32)
    tf = torch.zeros(11, 512)
    vol = ov.build_organ_volume(ct, model=object(), text_features=tf,
                                device="cpu", organs=[6, 7])
    assert vol.shape == (2, 4, 512, 512)          # 2 organs, D=4, native H,W
    assert np.allclose(vol[0], 6 / 10.0)          # organ 6 constant map
    assert np.allclose(vol[1], 7 / 10.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_organ_volume.py -q`
Expected: FAIL with `AttributeError: ... has no attribute 'build_organ_volume'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/tumor/organ_volume.py
import open_clip
from open_clip import get_tokenizer
from src.explainability.patch_similarity import BioMedCLIPPatchSimilarity
from src.training.freeze import UNFREEZE_PATTERNS
from src.labels.templates import CLASS_NAMES, FEW_SHOT_TEMPLATES

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

def load_organcam(checkpoint, device):
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)
    model = model.to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = get_tokenizer(MODEL_NAME)
    prompts = [FEW_SHOT_TEMPLATES[i] for i in range(len(CLASS_NAMES))]
    with torch.no_grad():
        tf = F.normalize(model.encode_text(tok(prompts).to(device)), dim=-1)
    return model, tf

def build_organ_volume(ct, model, text_features, device, organs=None):
    if organs is None:
        organs = list(range(len(CLASS_NAMES)))
    D, H, W = ct.shape
    out = np.zeros((len(organs), D, H, W), dtype=np.float32)
    explainer = BioMedCLIPPatchSimilarity(model, device=device)
    for zi in range(D):
        for oi, oc in enumerate(organs):
            img = preprocess_slice(ct[zi], oc).unsqueeze(0).to(device)
            m = explainer.generate(img, text_features, target_class=oc, image_size=224)
            t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
            out[oi, zi] = t.squeeze().numpy()
    explainer.remove_hooks()
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_organ_volume.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tumor/organ_volume.py tests/test_organ_volume.py
git commit -m "feat: build 3D organ-probability volume from OrganCAM per-slice"
```

---

### Task 8: The index — overlap, proximity, blend

**Files:**
- Create: `src/tumor/index.py`
- Test: `tests/test_index.py`

**Interfaces:**
- Produces:
  - `overlap(tumor: np.ndarray bool, organ: np.ndarray bool) -> float` — `|T∩O|/|T|`, `0.0` if tumor empty.
  - `proximity(tumor: np.ndarray bool, organ: np.ndarray bool, spacing: tuple) -> float` — `1/(1+d_mm)` where `d_mm` = min surface distance in mm via `scipy.ndimage.distance_transform_edt(~organ, sampling=spacing)` sampled at tumor voxels; `0.0` if either empty.
  - `blended_score(tumor, organ, spacing, alpha=0.5, beta=0.5) -> float` — `alpha*overlap + beta*proximity`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_index.py
import numpy as np
from src.tumor.index import overlap, proximity, blended_score

def _cube(shape, sl):
    m = np.zeros(shape, bool); m[sl] = True; return m

def test_overlap_full_and_none():
    T = _cube((10,10,10), np.s_[2:5,2:5,2:5])
    assert overlap(T, T) == 1.0
    O = _cube((10,10,10), np.s_[7:9,7:9,7:9])
    assert overlap(T, O) == 0.0

def test_overlap_partial():
    T = _cube((10,10,10), np.s_[0:4,0:1,0:1])   # 4 voxels
    O = _cube((10,10,10), np.s_[0:2,0:1,0:1])   # overlaps 2
    assert abs(overlap(T, O) - 0.5) < 1e-6

def test_proximity_touching_is_high():
    T = _cube((10,10,10), np.s_[0:1,0:1,0:1])
    O = _cube((10,10,10), np.s_[1:2,0:1,0:1])   # 1 voxel away
    p = proximity(T, O, spacing=(1.0,1.0,1.0))
    assert 0.4 < p < 0.6                          # d=1mm -> 1/(1+1)=0.5

def test_proximity_uses_spacing_mm():
    T = _cube((10,10,10), np.s_[0:1,0:1,0:1])
    O = _cube((10,10,10), np.s_[1:2,0:1,0:1])
    near = proximity(T, O, spacing=(0.5,0.5,0.5)) # 0.5mm -> 1/1.5
    far  = proximity(T, O, spacing=(4.0,4.0,4.0)) # 4mm  -> 1/5
    assert near > far

def test_proximity_empty_is_zero():
    Z = np.zeros((5,5,5), bool)
    assert proximity(Z, Z, (1,1,1)) == 0.0

def test_blended_score_weights():
    T = _cube((10,10,10), np.s_[0:2,0:1,0:1])
    O = _cube((10,10,10), np.s_[1:3,0:1,0:1])   # overlaps 1 of 2
    s = blended_score(T, O, (1,1,1), alpha=1.0, beta=0.0)
    assert abs(s - 0.5) < 1e-6                    # overlap only
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_index.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tumor.index'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/tumor/index.py
import numpy as np
from scipy.ndimage import distance_transform_edt

def overlap(tumor, organ):
    t = int(tumor.sum())
    if t == 0:
        return 0.0
    return float((tumor & organ).sum()) / float(t)

def proximity(tumor, organ, spacing):
    if not tumor.any() or not organ.any():
        return 0.0
    dist_mm = distance_transform_edt(~organ, sampling=spacing)
    d = float(dist_mm[tumor].min())
    return 1.0 / (1.0 + d)

def blended_score(tumor, organ, spacing, alpha=0.5, beta=0.5):
    return alpha * overlap(tumor, organ) + beta * proximity(tumor, organ, spacing)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_index.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/tumor/index.py tests/test_index.py
git commit -m "feat: overlap + spacing-aware proximity + blended index score"
```

---

### Task 9: Threshold organ volume + aggregate to a per-patient index

**Files:**
- Modify: `src/tumor/index.py`
- Test: `tests/test_index_aggregate.py`

**Interfaces:**
- Consumes: `blended_score` (Task 8); `CLASS_NAMES`.
- Produces:
  - `binarize_organs(organ_vol: np.ndarray[C,D,H,W], thresholds: dict[int,float], organs: list[int]) -> dict[int, np.ndarray bool]` — apply per-organ threshold (reuse `diagnostics/tuned_thresholds.json` values; default 0.0) to get boolean organ volumes keyed by class index.
  - `patient_index(tumor, organ_bools: dict[int,np.ndarray], spacing, alpha, beta) -> dict` — returns `{"per_organ": {name: {"overlap":..,"proximity":..,"score":..}}, "index": <max score over organs>}`. `index` = max blended score across organs (nearest/most-involved organ drives the patient score).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_index_aggregate.py
import numpy as np
from src.tumor.index import binarize_organs, patient_index

def test_binarize_applies_per_organ_threshold():
    vol = np.zeros((2, 3, 4, 4), np.float32)
    vol[0] = 0.2; vol[1] = 0.8
    b = binarize_organs(vol, thresholds={6: 0.5, 7: 0.5}, organs=[6, 7])
    assert b[6].sum() == 0 and b[7].all()

def test_patient_index_reports_max_and_per_organ():
    shape = (3, 4, 4)
    tumor = np.zeros(shape, bool); tumor[0, 0, 0] = True
    liver = np.zeros(shape, bool); liver[0, 0, 1] = True   # 1 voxel away
    kidney = np.zeros(shape, bool); kidney[2, 3, 3] = True # far
    res = patient_index(tumor, {6: liver, 4: kidney}, spacing=(1,1,1),
                        alpha=0.5, beta=0.5)
    assert "liver" in res["per_organ"] and "kidney-left" in res["per_organ"]
    assert res["index"] == max(o["score"] for o in res["per_organ"].values())
    assert res["per_organ"]["liver"]["score"] > res["per_organ"]["kidney-left"]["score"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_index_aggregate.py -q`
Expected: FAIL with `ImportError: cannot import name 'binarize_organs'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/tumor/index.py
from src.labels.templates import CLASS_NAMES

def binarize_organs(organ_vol, thresholds, organs):
    out = {}
    for oi, oc in enumerate(organs):
        thr = thresholds.get(oc, 0.0)
        out[oc] = organ_vol[oi] >= thr
    return out

def patient_index(tumor, organ_bools, spacing, alpha=0.5, beta=0.5):
    per_organ = {}
    for oc, ob in organ_bools.items():
        ov = overlap(tumor, ob)
        px = proximity(tumor, ob, spacing)
        per_organ[CLASS_NAMES[oc]] = {
            "overlap": ov, "proximity": px, "score": alpha * ov + beta * px,
        }
    index = max((o["score"] for o in per_organ.values()), default=0.0)
    return {"per_organ": per_organ, "index": index}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_index_aggregate.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tumor/index.py tests/test_index_aggregate.py
git commit -m "feat: binarize organ volume and aggregate per-patient metastasis index"
```

---

### Task 10: End-to-end index script (one patient)

**Files:**
- Create: `scripts/compute_index.py`
- Test: `tests/test_compute_index_smoke.py`

**Interfaces:**
- Consumes: everything above.
- Produces: CLI `python scripts/compute_index.py --root data/Spine-Mets-CT-SEG_v1_2024 --case <id> --checkpoint <ckpt> [--alpha 0.5 --beta 0.5 --limit_slices N]` → writes `results/bone_index/<case>.json` with the `patient_index` dict + metadata (spacing, tumor voxel count, alpha/beta). `--limit_slices` crops the CT to the tumor-bearing slice range ±N for speed.

- [ ] **Step 1: Write the failing test** (smoke: monkeypatch the heavy organ-volume builder)

```python
# tests/test_compute_index_smoke.py
import json, runpy, sys
import numpy as np
from pathlib import Path
import pytest
import scripts.compute_index as ci  # importable module

ROOT = Path("data/Spine-Mets-CT-SEG_v1_2024")

@pytest.mark.skipif(not (ROOT / "Spine-Mets-CT-SEG_Clinical.xlsx").exists(), reason="data absent")
def test_compute_index_writes_json(tmp_path, monkeypatch):
    # stub the model + organ volume so no GPU/network is needed
    monkeypatch.setattr(ci, "load_organcam", lambda ckpt, device: (object(), None))
    def fake_vol(ct, model, tf, device, organs=None):
        organs = organs or list(range(11))
        v = np.zeros((len(organs), *ct.shape), np.float32)
        v[:] = 1.0   # every organ "present" everywhere -> nonzero index
        return v
    monkeypatch.setattr(ci, "build_organ_volume", fake_vol)
    out = ci.run(root=str(ROOT), case=None, checkpoint="x", device="cpu",
                 alpha=0.5, beta=0.5, limit_slices=8, out_dir=str(tmp_path))
    data = json.loads(Path(out).read_text())
    assert "index" in data and "per_organ" in data
    assert data["tumor_voxels"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_compute_index_smoke.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.compute_index'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/compute_index.py
"""Compute the bone-metastasis organ index for one Spine-Mets patient."""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.spine_mets_labels import load_lesion_labels
from src.data.spine_mets import load_patient
from src.tumor.organ_volume import load_organcam, build_organ_volume
from src.tumor.index import binarize_organs, patient_index
from src.labels.templates import CLASS_NAMES

DEFAULT_CKPT = "checkpoints/totalseg_patch_v5/epoch_011_val3.4584.pt"

def _load_thresholds():
    p = Path("diagnostics/tuned_thresholds.json")
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    name_to_idx = {v: k for k, v in CLASS_NAMES.items()}
    out = {}
    for name, v in data.items():
        if name in name_to_idx and isinstance(v, (int, float)):
            out[name_to_idx[name]] = float(v)
    return out

def run(root, case, checkpoint, device, alpha, beta, limit_slices, out_dir):
    labels = load_lesion_labels(str(Path(root) / "Spine-Mets-CT-SEG_Clinical.xlsx"))
    if case is None:
        cases_dir = Path(root) / "spine_mets_ct_seg"
        case = next(c.name for c in sorted(cases_dir.iterdir())
                    if c.is_dir() and labels.get(c.name))
    p = load_patient(root, case, labels)
    # crop to tumor-bearing slice range for speed
    zs = np.where(p.tumor_mask.any(axis=(1, 2)))[0]
    z0, z1 = max(0, zs.min() - limit_slices), min(p.ct.shape[0], zs.max() + limit_slices + 1)
    ct = p.ct[z0:z1]; tumor = p.tumor_mask[z0:z1]
    organs = list(range(len(CLASS_NAMES)))
    model, tf = load_organcam(checkpoint, device)
    organ_vol = build_organ_volume(ct, model, tf, device, organs)
    organ_bools = binarize_organs(organ_vol, _load_thresholds(), organs)
    res = patient_index(tumor, organ_bools, p.spacing, alpha, beta)
    res.update(case=case, spacing=list(p.spacing), tumor_voxels=int(tumor.sum()),
               alpha=alpha, beta=beta, slice_range=[int(z0), int(z1)])
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case}.json"
    out_path.write_text(json.dumps(res, indent=2))
    return str(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Spine-Mets-CT-SEG_v1_2024")
    ap.add_argument("--case", default=None)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--limit_slices", type=int, default=8)
    ap.add_argument("--out_dir", default="results/bone_index")
    a = ap.parse_args()
    out = run(a.root, a.case, a.checkpoint, a.device, a.alpha, a.beta,
              a.limit_slices, a.out_dir)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
```

Also create `scripts/__init__.py` (empty) if it does not exist, so `import scripts.compute_index` works in the test.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_compute_index_smoke.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/compute_index.py scripts/__init__.py tests/test_compute_index_smoke.py
git commit -m "feat: end-to-end per-patient bone-metastasis index script"
```

---

### Task 11: Real GPU smoke run + qualitative overlay

**Files:**
- Create: `scripts/visualize_index.py`
- Test: manual (documented below — requires GPU + model weights)

**Interfaces:**
- Consumes: `load_patient`, `load_organcam`, `build_organ_volume`.
- Produces: `scripts/visualize_index.py --case <id>` → saves `figures/bone_index/<case>_overlay.png`: for the tumor's peak slice, CT with tumor mask (red) + top-scoring organ map (heat) overlaid.

- [ ] **Step 1: Real one-patient run (manual verification)**

Run: `.venv/Scripts/python.exe scripts/compute_index.py --case 10355 --device cuda --limit_slices 6`
Expected: prints `wrote results/bone_index/10355.json`; JSON has `index` in `[0,1]`, nonzero `tumor_voxels`, and `per_organ` scores where anatomically-adjacent organs (e.g. lung at thoracic levels) score higher than distant ones.

- [ ] **Step 2: Write the overlay visualizer**

```python
# scripts/visualize_index.py
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.spine_mets_labels import load_lesion_labels
from src.data.spine_mets import load_patient
from src.tumor.organ_volume import load_organcam, build_organ_volume
from src.labels.templates import CLASS_NAMES

DEFAULT_CKPT = "checkpoints/totalseg_patch_v5/epoch_011_val3.4584.pt"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Spine-Mets-CT-SEG_v1_2024")
    ap.add_argument("--case", required=True)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    labels = load_lesion_labels(str(Path(a.root) / "Spine-Mets-CT-SEG_Clinical.xlsx"))
    p = load_patient(a.root, a.case, labels)
    z = int(np.argmax(p.tumor_mask.sum(axis=(1, 2))))     # peak tumor slice
    model, tf = load_organcam(a.checkpoint, a.device)
    vol = build_organ_volume(p.ct[z:z+1], model, tf, a.device)  # (11,1,H,W)
    top = int(np.argmax([vol[o, 0][p.tumor_mask[z]].mean() if p.tumor_mask[z].any()
                         else vol[o, 0].mean() for o in range(11)]))
    ct = np.clip(p.ct[z], -150, 250); ct = (ct + 150) / 400
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(ct, cmap="gray")
    ax[0].imshow(np.ma.masked_where(~p.tumor_mask[z], p.tumor_mask[z]), cmap="autumn", alpha=0.7)
    ax[0].set_title(f"{a.case}: tumor (slice {z})"); ax[0].axis("off")
    ax[1].imshow(ct, cmap="gray")
    ax[1].imshow(vol[top, 0], cmap="jet", alpha=0.5)
    ax[1].set_title(f"top organ map: {CLASS_NAMES[top]}"); ax[1].axis("off")
    out = Path("figures/bone_index"); out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out / f"{a.case}_overlay.png", dpi=110)
    print(f"wrote {out / (a.case + '_overlay.png')}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the visualizer**

Run: `.venv/Scripts/python.exe scripts/visualize_index.py --case 10355 --device cuda`
Expected: `wrote figures/bone_index/10355_overlay.png`; opening it shows the tumor over the vertebra and an anatomically-plausible organ heatmap.

- [ ] **Step 4: Commit**

```bash
git add scripts/visualize_index.py results/bone_index/ figures/bone_index/
git commit -m "feat: qualitative overlay + first real bone-index run"
```

---

### Task 12: Organ-transfer sanity check on CT-ORG

**Files:**
- Create: `scripts/eval_organ_transfer.py`
- Test: `tests/test_ctorg_loader.py`

**Interfaces:**
- Consumes: `preprocess_slice`, `load_organcam`, `build_organ_volume`; `nibabel` for CT-ORG NIfTI.
- Produces:
  - `src/data/ct_org.py::load_ctorg_case(vol_path, label_path) -> tuple[np.ndarray ct, np.ndarray labels, spacing]` — CT-ORG stores labels as integer maps (0 bg, 1 liver, 2 bladder, 3 lungs, 4 kidneys, 5 bone per its README); loader returns HU volume + integer label volume.
  - `scripts/eval_organ_transfer.py` — run OrganCAM organ maps on a few CT-ORG volumes, threshold, compute IoU vs CT-ORG organ GT for the organs both datasets share (liver, kidney, lung, bladder). Confirms OrganCAM transfers to non-TotalSegmentator CT before the index is trusted.

- [ ] **Step 1: Write the failing test** (loader only — no model)

```python
# tests/test_ctorg_loader.py
from pathlib import Path
import numpy as np
import pytest
from src.data.ct_org import load_ctorg_case

CTORG = Path("data/CT-ORG")

@pytest.mark.skipif(not CTORG.exists(), reason="CT-ORG absent")
def test_load_ctorg_case():
    vols = sorted(CTORG.glob("**/volume-*.nii.gz"))
    labs = sorted(CTORG.glob("**/labels-*.nii.gz"))
    assert vols and labs, "expected CT-ORG volume-*/labels-* NIfTI files"
    ct, lab, spacing = load_ctorg_case(str(vols[0]), str(labs[0]))
    assert ct.shape == lab.shape and len(spacing) == 3
    assert ct.min() < -500 and ct.max() > 500       # HU
    assert set(np.unique(lab)).issubset(set(range(6)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ctorg_loader.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.ct_org'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/ct_org.py
import nibabel as nib
import numpy as np

def load_ctorg_case(vol_path, label_path):
    v = nib.load(vol_path)
    ct = np.asarray(v.dataobj).astype(np.float32)
    lab = np.asarray(nib.load(label_path).dataobj).astype(np.int16)
    zoom = v.header.get_zooms()
    spacing = (float(zoom[2]), float(zoom[1]), float(zoom[0]))  # z,y,x
    # NIfTI is x,y,z; transpose to z,y,x to match our volume convention
    ct = np.transpose(ct, (2, 1, 0))
    lab = np.transpose(lab, (2, 1, 0))
    return ct, lab, spacing
```

Then `scripts/eval_organ_transfer.py` (run manually, no unit test — needs GPU):

```python
# scripts/eval_organ_transfer.py
import argparse, sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.ct_org import load_ctorg_case
from src.tumor.organ_volume import load_organcam, build_organ_volume
from src.tumor.index import binarize_organs
from src.labels.templates import CLASS_NAMES

# CT-ORG label int -> OrganCAM class indices (liver, lungs L+R, kidneys L+R, bladder)
CTORG_TO_ORGANCAM = {1: [6], 3: [7, 8], 4: [4, 5], 2: [0]}
DEFAULT_CKPT = "checkpoints/totalseg_patch_v5/epoch_011_val3.4584.pt"

def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum()) / float(u) if u else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT); ap.add_argument("--device", default="cuda")
    ap.add_argument("--stride", type=int, default=4)  # subsample slices for speed
    a = ap.parse_args()
    ct, lab, _ = load_ctorg_case(a.vol, a.labels)
    ct, lab = ct[::a.stride], lab[::a.stride]
    model, tf = load_organcam(a.checkpoint, a.device)
    organs = sorted({o for lst in CTORG_TO_ORGANCAM.values() for o in lst})
    vol = build_organ_volume(ct, model, tf, a.device, organs)
    ob = binarize_organs(vol, {}, organs)   # threshold 0.0 (raw sims)
    results = {}
    for ctorg_lbl, oc_list in CTORG_TO_ORGANCAM.items():
        gt = lab == ctorg_lbl
        pred = np.zeros_like(gt)
        for oc in oc_list:
            pred |= ob[oc]
        results["+".join(CLASS_NAMES[o] for o in oc_list)] = iou(pred, gt)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run loader test + manual transfer check**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ctorg_loader.py -q`
Expected: PASS
Then (manual): `.venv/Scripts/python.exe scripts/eval_organ_transfer.py --vol <ctorg volume> --labels <ctorg labels> --device cuda` → prints per-organ IoU; liver/lung expected > 0.4 if transfer holds.

- [ ] **Step 5: Commit**

```bash
git add src/data/ct_org.py scripts/eval_organ_transfer.py tests/test_ctorg_loader.py
git commit -m "feat: CT-ORG loader + OrganCAM organ-transfer sanity check"
```

---

### Task 13: Batch over all patients + summary; README section

**Files:**
- Create: `scripts/batch_index.py`
- Modify: `README.md` (add "Bone-metastasis organ index (Phase 1)" section)
- Test: `tests/test_batch_index.py`

**Interfaces:**
- Consumes: `compute_index.run` (Task 10).
- Produces: `batch_index.py --root ... --checkpoint ...` iterates every case with lesion labels, calls `compute_index.run`, and writes `results/bone_index/summary.json` = `{case: index}` plus mean/median. `aggregate_summary(per_case: dict[str,float]) -> dict` is the unit-tested pure helper.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_batch_index.py
from scripts.batch_index import aggregate_summary

def test_aggregate_summary_stats():
    s = aggregate_summary({"a": 0.2, "b": 0.8, "c": 0.5})
    assert s["n"] == 3
    assert abs(s["mean"] - 0.5) < 1e-6
    assert abs(s["median"] - 0.5) < 1e-6
    assert s["min"] == 0.2 and s["max"] == 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_batch_index.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.batch_index'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/batch_index.py
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.spine_mets_labels import load_lesion_labels
import scripts.compute_index as ci

def aggregate_summary(per_case):
    vals = np.array(list(per_case.values()), dtype=float)
    return {"n": int(vals.size), "mean": float(vals.mean()), "median": float(np.median(vals)),
            "min": float(vals.min()), "max": float(vals.max())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Spine-Mets-CT-SEG_v1_2024")
    ap.add_argument("--checkpoint", default=ci.DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=0.5); ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--limit_slices", type=int, default=8)
    ap.add_argument("--out_dir", default="results/bone_index")
    a = ap.parse_args()
    labels = load_lesion_labels(str(Path(a.root) / "Spine-Mets-CT-SEG_Clinical.xlsx"))
    cases_dir = Path(a.root) / "spine_mets_ct_seg"
    per_case = {}
    for c in sorted(cases_dir.iterdir()):
        if not (c.is_dir() and labels.get(c.name)):
            continue
        try:
            out = ci.run(a.root, c.name, a.checkpoint, a.device, a.alpha, a.beta,
                         a.limit_slices, a.out_dir)
            per_case[c.name] = json.loads(Path(out).read_text())["index"]
            print(f"{c.name}: {per_case[c.name]:.3f}", flush=True)
        except Exception as e:
            print(f"{c.name}: SKIP ({e})", flush=True)
    summary = {"per_case": per_case, **aggregate_summary(per_case)}
    Path(a.out_dir, "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"mean index {summary['mean']:.3f} over {summary['n']} patients")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test + full batch**

Run: `.venv/Scripts/python.exe -m pytest tests/test_batch_index.py -q`
Expected: PASS
Then (manual, GPU): `.venv/Scripts/python.exe scripts/batch_index.py --device cuda` → writes `results/bone_index/summary.json`.

- [ ] **Step 5: Add README section + commit**

Add to `README.md` a "## Bone-metastasis organ index (Phase 1)" section describing: datasets (Spine-Mets anchor, CT-ORG validation), the blended overlap+proximity index, the `compute_index.py` / `batch_index.py` / `eval_organ_transfer.py` commands, and that tumor GT comes from Spine-Mets lesion labels (Phase 2 will make tumor localization weakly-supervised).

```bash
git add scripts/batch_index.py tests/test_batch_index.py README.md results/bone_index/summary.json
git commit -m "feat: batch bone-index over all patients + README section"
```

---

## Self-Review

**Spec coverage:**
- §3 datasets → Tasks 2–5 (Spine-Mets + clinical labels), 12 (CT-ORG). ✓
- §4.1 tumor GT (lesion ∩ vertebrae) → Task 5. ✓ (skeleton HU-threshold from spec §5 `src/tumor/skeleton.py` is redundant for Spine-Mets since vertebral SEG is already bone — deferred to Phase 2/BM-Seg where lesion masks aren't vertebra-constrained; noted here rather than adding an unused module.)
- §4.2 organ maps, 2D→3D stack → Tasks 6–7. ✓
- §4.3 blended overlap+proximity + aggregation → Tasks 8–9. ✓
- §7 validation: organ transfer → Task 12; index run → Tasks 11, 13. ✓
- §5 architecture module layout → matches (`spine_mets.py`, `organ_volume.py`, `index.py`, scripts). `organ_volume.py` absorbs the `build_organ_volume.py` script role via `scripts/compute_index.py`. ✓

**Deviations from spec (intentional):** `src/tumor/skeleton.py` is not created in Phase 1 — the Spine-Mets tumor mask is already bone-constrained (vertebral SEG), so an HU skeleton adds nothing here. It belongs to Phase 2 (BM-Seg / weakly-supervised masks). Flagged so a reviewer expecting it knows why.

**Placeholder scan:** none — every code step contains runnable code; every run step has an exact command + expected output.

**Type consistency:** `load_patient` returns `SpineMetsPatient` (Task 5) consumed by Tasks 10–11; `build_organ_volume` returns `[C,D,H,W]` (Task 7) consumed by `binarize_organs` (Task 9) and scripts; `patient_index` dict shape (Task 9) consumed by `compute_index.run` (Task 10) and `batch_index` (Task 13). Consistent.

## Open parameters (resolve empirically during execution)

- **α/β blend weights** — default 0.5/0.5; sweep once real per-patient scores exist (Task 13) and pick values that separate high- vs low-burden patients.
- **Patient aggregation** — Task 9 uses max-over-organs; alternative burden-weighted sum is a one-line change if the max proves too sensitive to a single adjacency.
- **Organ thresholds** — reuses `diagnostics/tuned_thresholds.json`; may need re-tuning on Spine-Mets if the CT-ORG transfer check (Task 12) shows drift.
