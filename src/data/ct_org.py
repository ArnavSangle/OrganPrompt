"""Load CT-ORG cases: HU volume + integer organ-label volume, aligned (z,y,x)."""
import nibabel as nib
import numpy as np


def load_ctorg_case(vol_path, label_path):
    """CT-ORG labels: 0 bg, 1 liver, 2 bladder, 3 lungs, 4 kidneys, 5 bone."""
    v = nib.load(vol_path)
    ct = np.asarray(v.dataobj).astype(np.float32)
    lab = np.asarray(nib.load(label_path).dataobj).astype(np.int16)
    zoom = v.header.get_zooms()
    spacing = (float(zoom[2]), float(zoom[1]), float(zoom[0]))  # z,y,x
    # NIfTI is x,y,z; transpose to z,y,x to match our volume convention
    ct = np.transpose(ct, (2, 1, 0))
    lab = np.transpose(lab, (2, 1, 0))
    return ct, lab, spacing
