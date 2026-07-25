"""Bone-metastasis-to-organ index: overlap, physical-distance proximity, blend."""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from src.labels.templates import CLASS_NAMES


def overlap(tumor, organ):
    """Fraction of tumor voxels also inside organ: |T∩O|/|T|."""
    t = int(tumor.sum())
    if t == 0:
        return 0.0
    return float((tumor & organ).sum()) / float(t)


def min_distance_mm(tumor_coords, organ_coords, spacing):
    """Min physical distance (mm) between two voxel-coordinate sets, or inf if empty.

    Deliberately NOT distance_transform_edt. The EDT allocates a float64 array over
    the *whole* volume just to read a single minimum out of it -- ~600MB per organ
    on a full-resolution scan, times 11 organs, which is enough to exhaust system
    memory. Both point sets here are tiny (select_peak_component caps an organ at
    the top 0.1% of voxels), so a KD-tree over the points answers the same question
    in a few MB.
    """
    if len(tumor_coords) == 0 or len(organ_coords) == 0:
        return float("inf")
    sp = np.asarray(spacing, dtype=np.float64)
    tree = cKDTree(np.asarray(organ_coords, dtype=np.float64) * sp)
    d, _ = tree.query(np.asarray(tumor_coords, dtype=np.float64) * sp, k=1)
    return float(d.min())


def proximity(tumor, organ, spacing):
    """1/(1+d_mm), d_mm = min physical surface distance from tumor to organ."""
    if not tumor.any() or not organ.any():
        return 0.0
    d = min_distance_mm(np.argwhere(tumor), np.argwhere(organ), spacing)
    return 1.0 / (1.0 + d) if np.isfinite(d) else 0.0


def blended_score(tumor, organ, spacing, alpha=0.5, beta=0.5):
    return alpha * overlap(tumor, organ) + beta * proximity(tumor, organ, spacing)


def binarize_organs(organ_vol, thresholds, organs):
    """Apply a per-organ threshold to a [organs,D,H,W] similarity volume."""
    out = {}
    for oi, oc in enumerate(organs):
        thr = thresholds.get(oc, 0.0)
        out[oc] = organ_vol[oi] >= thr
    return out


def denoise_organ_mask(mask, min_size=500):
    """Drop connected components smaller than min_size, per axial slice.

    Per-slice organ classification carries speckle false positives (a handful of
    scattered voxels here and there). Over a large tumor's surface these are enough
    to sit "adjacent" purely by chance, making proximity ~1 regardless of true
    anatomical distance. Small components are almost always this noise, not organ
    tissue, so drop them before any distance/overlap computation.
    """
    out = np.zeros_like(mask, dtype=bool)
    for zi in range(mask.shape[0]):
        sl = mask[zi]
        if not sl.any():
            continue
        labeled, n = ndimage.label(sl)
        sizes = ndimage.sum(sl, labeled, range(1, n + 1))
        keep = np.zeros_like(sl, dtype=bool)
        for i, sz in enumerate(sizes):
            if sz >= min_size:
                keep |= labeled == (i + 1)
        out[zi] = keep
    return out


def select_peak_component(sim, percentile=99.9):
    """Binarize at a strict per-organ percentile, keep only the 3D-connected
    component containing the single highest-similarity voxel.

    Absolute per-organ thresholds (tuned on TotalSegmentator's single-organ-per-image
    classification) are far too permissive when scanning every organ across every
    slice of a much larger volume: dense, low-level false-positive coverage lets
    small blobs land immediately adjacent to a large tumor by pure chance, regardless
    of true anatomical distance. Empirically, the single global-max-similarity voxel
    per organ is reliably near the true organ location; a strict percentile cut
    breaks the raw similarity map into disconnected pieces, and keeping only the
    piece containing that peak discards everything else as noise.
    """
    thr = np.percentile(sim, percentile)
    mask = sim >= thr
    if mask.all():
        return np.zeros_like(mask, dtype=bool)   # flat/uniform input, no real signal
    peak = np.unravel_index(np.argmax(sim), sim.shape)
    labeled, n = ndimage.label(mask)
    seed_label = labeled[peak]
    if seed_label == 0:
        return np.zeros_like(mask, dtype=bool)
    return labeled == seed_label


def patient_index(tumor, organ_bools, spacing, alpha=0.5, beta=0.5):
    """Blend overlap+proximity per organ; index = max score (nearest/most-involved organ)."""
    per_organ = {}
    for oc, ob in organ_bools.items():
        ov = overlap(tumor, ob)
        px = proximity(tumor, ob, spacing)
        per_organ[CLASS_NAMES[oc]] = {
            "overlap": ov, "proximity": px, "score": alpha * ov + beta * px,
        }
    index = max((o["score"] for o in per_organ.values()), default=0.0)
    return {"per_organ": per_organ, "index": index}
