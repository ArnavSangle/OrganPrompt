"""Bone-metastasis-to-organ index: overlap, physical-distance proximity, blend."""
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

from src.labels.templates import CLASS_NAMES


def overlap(tumor, organ):
    """Fraction of tumor voxels also inside organ: |T∩O|/|T|."""
    t = int(tumor.sum())
    if t == 0:
        return 0.0
    return float((tumor & organ).sum()) / float(t)


def proximity(tumor, organ, spacing):
    """1/(1+d_mm), d_mm = min physical surface distance from tumor to organ."""
    if not tumor.any() or not organ.any():
        return 0.0
    dist_mm = distance_transform_edt(~organ, sampling=spacing)
    d = float(dist_mm[tumor].min())
    return 1.0 / (1.0 + d)


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
