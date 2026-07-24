"""Bone-metastasis-to-organ index: overlap, physical-distance proximity, blend."""
import numpy as np
from scipy.ndimage import distance_transform_edt


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
