"""Build a 3D organ-probability volume by running OrganCAM per axial slice.

Bridges OrganCAM's inherently-2D patch-similarity inference (224x224, ViT-B/16)
to the 3D geometry the metastasis index needs: each CT slice is preprocessed
exactly as in the OrganCAM training/eval pipeline, run through
BioMedCLIPPatchSimilarity, and the resulting per-slice organ maps are stacked
and downsampled back to the native slice resolution.
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.data.totalsegmentator import HU_WINDOW


def preprocess_slice(hu_slice, organ_class):
    """HU-window (per organ), normalize to [0,1], resize to 224, replicate to 3ch.

    Mirrors TotalSegmentatorDataset's slow-path preprocessing
    (src/data/totalsegmentator.py) so OrganCAM sees the same input distribution
    it was trained on.
    """
    lo, hi = HU_WINDOW.get(organ_class, HU_WINDOW["default"])
    x = np.clip(hu_slice.astype(np.float32), lo, hi)
    x = (x - lo) / (hi - lo)
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)          # (1,1,H,W)
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    return t.squeeze(0).repeat(3, 1, 1)                        # (3,224,224)
