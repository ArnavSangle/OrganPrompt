"""Load Spine-Mets-CT-SEG patients: CT volume, per-vertebra masks, tumor mask.

DICOM layout per patient:
    <case>/<study>/<series>/*.dcm
where one CT series (``SKIN_TO_SKIN_SIM...``, Modality=CT, most slices) holds the
axial stack and one SEG file (Modality=SEG) holds per-vertebra binary masks —
one segment per vertebral level, labelled e.g. ``"T1 vertebra"``.

The SEG is parsed with plain pydicom (pydicom_seg is incompatible with
pydicom 3.x): each frame carries a ``SegmentIdentificationSequence`` (which
vertebra) and a ``PlanePositionSequence`` (physical z), matched to the CT stack
by z-position.
"""
import glob
import os
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pydicom


def find_series(patient_dir):
    """Return (ct_dicom_paths_sorted_by_z, seg_dicom_path)."""
    dcms = glob.glob(os.path.join(patient_dir, "**", "*.dcm"), recursive=True)
    seg_path = None
    ct_candidates = []
    for p in dcms:
        d = pydicom.dcmread(p, stop_before_pixels=True)
        modality = getattr(d, "Modality", None)
        if modality == "SEG":
            seg_path = p
        elif modality == "CT":
            ct_candidates.append(p)
    if seg_path is None or not ct_candidates:
        raise FileNotFoundError(f"CT or SEG series missing in {patient_dir}")
    by_dir = Counter(os.path.dirname(p) for p in ct_candidates)
    ct_dir = by_dir.most_common(1)[0][0]
    ct = [p for p in ct_candidates if os.path.dirname(p) == ct_dir]
    ct.sort(key=lambda p: float(
        pydicom.dcmread(p, stop_before_pixels=True).ImagePositionPatient[2]))
    return ct, seg_path


def load_ct_volume(ct_paths):
    """Return (hu_volume[D,H,W] float32, spacing_zyx_mm)."""
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
    """'T1 vertebra' -> 'T1'; None if not a vertebra segment."""
    tok = str(label).split()[0].upper()
    return tok if len(tok) >= 2 and tok[0] in "CTLS" and tok[1:].isdigit() else None


def load_vertebra_masks(seg_path, ct_paths):
    """Return {level: bool_mask[D,H,W]} placed on the CT grid by z-position."""
    seg = pydicom.dcmread(seg_path)
    frames = seg.pixel_array
    if frames.ndim == 2:
        frames = frames[None]
    H, W = int(seg.Rows), int(seg.Columns)
    ct_z = np.array([
        float(pydicom.dcmread(p, stop_before_pixels=True).ImagePositionPatient[2])
        for p in ct_paths
    ])
    D = len(ct_paths)
    seg_level = {int(s.SegmentNumber): _level_from_label(s.SegmentLabel)
                 for s in seg.SegmentSequence}
    masks = {}
    for f_idx, fg in enumerate(seg.PerFrameFunctionalGroupsSequence):
        seg_num = int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        level = seg_level.get(seg_num)
        if level is None:
            continue
        zpos = float(fg.PlanePositionSequence[0].ImagePositionPatient[2])
        z = int(np.argmin(np.abs(ct_z - zpos)))
        if abs(ct_z[z] - zpos) > 2.0:            # mm tolerance
            continue
        if level not in masks:
            masks[level] = np.zeros((D, H, W), dtype=bool)
        masks[level][z] |= frames[f_idx].astype(bool)
    return {lvl: m for lvl, m in masks.items() if m.any()}


_TYPE_CODE = {"lytic": 1, "blastic": 2, "mixed": 3}


def stride_coords(coords, stride):
    """Map full-volume voxel coordinates onto a z-subsampled volume.

    ct[::stride] keeps slices 0, stride, 2*stride..., so a lesion's coordinates
    have to be filtered to those slices and then renumbered to the new index
    space. Skipping the renumber leaves every lesion at its original z and
    silently multiplies its distance to everything by the stride.

    A thin lesion can survive on none of the retained slices and come back empty;
    the caller has to notice rather than score an empty lesion.
    """
    coords = np.asarray(coords)
    if stride <= 1 or len(coords) == 0:
        return coords
    kept = coords[coords[:, 0] % stride == 0].copy()
    kept[:, 0] //= stride
    return kept


@dataclass
class SpineMetsPatient:
    case: str
    ct: np.ndarray               # [D,H,W] HU float32
    spacing: tuple               # (dz, dy, dx) mm
    tumor_mask: np.ndarray       # [D,H,W] bool
    lesion_type: np.ndarray      # [D,H,W] int8 (0 none,1 lytic,2 blastic,3 mixed)
    # {level: [N,3] voxel coords}, one entry per LABELLED level. Coordinates, not
    # masks: a patient can carry 17 labelled levels and 17 volume-sized boolean
    # masks is over a gigabyte, while the coordinates are a few megabytes. Their
    # union is exactly tumor_mask.
    lesion_coords: dict = field(default_factory=dict)


def load_patient(root, case, labels):
    """Assemble a patient: tumor mask = union of vertebra masks for labelled levels."""
    patient_dir = os.path.join(root, "spine_mets_ct_seg", case)
    ct_paths, seg_path = find_series(patient_dir)
    ct, spacing = load_ct_volume(ct_paths)
    vmasks = load_vertebra_masks(seg_path, ct_paths)
    tumor = np.zeros(ct.shape, dtype=bool)
    ltype = np.zeros(ct.shape, dtype=np.int8)
    lesions = {}
    for level, kind in labels.get(case, {}).items():
        m = vmasks.get(level)
        if m is None:
            continue
        tumor |= m
        ltype[m] = _TYPE_CODE[kind]
        lesions[level] = np.argwhere(m)
    return SpineMetsPatient(case, ct, spacing, tumor, ltype, lesions)
