"""How much anatomy a Spine-Mets scan actually contains.

The per-lesion validation leaves 38 anatomically impossible top-1 detections,
and bladder plus femur account for 24 of them -- 15 at thoracic levels. Because
proximity is 1/(1+d), a distant organ cannot outrank a close one; a bladder that
wins a T4 lesion from 12mm away is a bladder-labelled mask sitting in the chest,
not a real bladder being mis-scored from the pelvis.

That leaves one question worth answering before any fix is written: was the
pelvis in the imaged volume at all? These are SKIN_TO_SKIN_SIM radiotherapy
planning scans, so some reach the femurs and some stop at the diaphragm.

  - pelvis NOT in view -> the organ cannot be present, and saying so is physics
    rather than a prior tuned to the evaluation set
  - pelvis in view      -> a real bladder lost to its own phantom on peak
    response, which is an assignment problem instead

Everything here reads DICOM headers only, never pixel data, so it costs seconds
per patient and needs no GPU.

The SEG carries only the vertebrae somebody annotated, so its span is a LOWER
bound on the imaged anatomy, never an equality. That is why caudal_margin_mm
exists: the unlabelled millimetres below the last segmented vertebra are the
part of the scan the segmentation says nothing about.
"""
import pydicom

from src.data.spine_mets import find_series, _level_from_label

LEVEL_ORDER = ([f"C{i}" for i in range(1, 8)]
               + [f"T{i}" for i in range(1, 13)]
               + [f"L{i}" for i in range(1, 6)]
               + [f"S{i}" for i in range(1, 6)])
_RANK = {lvl: i for i, lvl in enumerate(LEVEL_ORDER)}


def level_rank(level):
    """Craniocaudal index, or None if the label is not a vertebra.

    Sorting the level strings lexically would put L1 above T1 and invert the
    whole question, so ordering always goes through this table.
    """
    return _RANK.get(str(level).strip().upper())


def lowest_level(levels):
    """The most caudal recognised vertebra, ignoring non-vertebral segments."""
    ranked = [(r, lvl) for lvl in levels if (r := level_rank(lvl)) is not None]
    return max(ranked)[1] if ranked else None


def caudal_margin_mm(vert_z, z_min):
    """Millimetres of imaged anatomy below the last segmented vertebra.

    None when nothing was segmented -- an absent measurement, which must not be
    confused with a measured zero.
    """
    if not vert_z:
        return None
    return min(lo for lo, _ in vert_z.values()) - z_min


def vertebra_z_extent(seg_path):
    """{level: (z_min_mm, z_max_mm)} from SEG headers, without reading pixels."""
    seg = pydicom.dcmread(seg_path, stop_before_pixels=True)
    seg_level = {int(s.SegmentNumber): _level_from_label(s.SegmentLabel)
                 for s in seg.SegmentSequence}
    out = {}
    for fg in seg.PerFrameFunctionalGroupsSequence:
        num = int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        level = seg_level.get(num)
        if level is None:
            continue
        z = float(fg.PlanePositionSequence[0].ImagePositionPatient[2])
        lo, hi = out.get(level, (z, z))
        out[level] = (min(lo, z), max(hi, z))
    return out


def ct_z_bounds(ct_paths):
    """(z_min_mm, z_max_mm, n_slices) from CT headers, without reading pixels."""
    zs = [float(pydicom.dcmread(p, stop_before_pixels=True).ImagePositionPatient[2])
          for p in ct_paths]
    return min(zs), max(zs), len(zs)


def scan_extent(patient_dir):
    """Craniocaudal coverage of one patient, from headers alone."""
    ct_paths, seg_path = find_series(patient_dir)
    z_min, z_max, n = ct_z_bounds(ct_paths)
    vz = vertebra_z_extent(seg_path)
    return {
        "levels": sorted(vz, key=level_rank),
        "lowest_level": lowest_level(vz),
        "highest_level": (sorted(vz, key=level_rank)[0] if vz else None),
        "z_min_mm": z_min,
        "z_max_mm": z_max,
        "n_slices": n,
        "length_mm": z_max - z_min,
        "caudal_margin_mm": caudal_margin_mm(vz, z_min),
    }
