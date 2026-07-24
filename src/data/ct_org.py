"""Load CT-ORG cases: HU volume + integer organ-label volume, aligned (z,x,y)."""
import nibabel as nib
import numpy as np


def load_ctorg_case(vol_path, label_path):
    """CT-ORG labels: 0 bg, 1 liver, 2 bladder, 3 lungs, 4 kidneys, 5 bone.

    NIfTI arrays are (X,Y,Z). We move Z to the front for a per-slice (D,H,W)
    volume, but must NOT swap X/Y while doing so: OrganCAM's few-shot prompts
    encode absolute in-slice position ("top half", "right side"), which was
    trained against TotalSegmentator's convention of slicing ct_vol[:, :, z]
    and using the (X,Y) plane untouched. Swapping X/Y here would silently
    rotate every slice relative to that convention and desync organ position
    from the text prompts.
    """
    v = nib.load(vol_path)
    ct = np.asarray(v.dataobj).astype(np.float32)
    lab = np.asarray(nib.load(label_path).dataobj).astype(np.int16)
    zoom = v.header.get_zooms()
    spacing = (float(zoom[2]), float(zoom[0]), float(zoom[1]))  # z,x,y
    ct = np.transpose(ct, (2, 0, 1))
    lab = np.transpose(lab, (2, 0, 1))
    return ct, lab, spacing
