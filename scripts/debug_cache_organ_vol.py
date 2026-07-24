"""Ad hoc: cache one patient's raw organ-similarity volume + tumor mask to .npz,
so denoise/threshold parameters can be swept offline on CPU without re-running
the model each time."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.spine_mets_labels import load_lesion_labels
from src.data.spine_mets import load_patient
from src.tumor.organ_volume import load_organcam, build_organ_volume
from src.labels.templates import CLASS_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Spine-Mets-CT-SEG_v1_2024")
    ap.add_argument("--case", required=True)
    ap.add_argument("--checkpoint", default="checkpoints/totalseg_patch_v5/epoch_011_val3.4584.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit_slices", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    labels = load_lesion_labels(str(Path(a.root) / "Spine-Mets-CT-SEG_Clinical.xlsx"))
    p = load_patient(a.root, a.case, labels)
    zs = np.where(p.tumor_mask.any(axis=(1, 2)))[0]
    z0 = max(0, zs.min() - a.limit_slices)
    z1 = min(p.ct.shape[0], zs.max() + a.limit_slices + 1)
    ct = p.ct[z0:z1]
    tumor = p.tumor_mask[z0:z1]
    organs = list(range(len(CLASS_NAMES)))
    model, tf = load_organcam(a.checkpoint, a.device)
    organ_vol = build_organ_volume(ct, model, tf, a.device, organs)

    out = a.out or f"results/bone_index/{a.case}_cached.npz"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, organ_vol=organ_vol, tumor=tumor, spacing=np.array(p.spacing))
    print(f"wrote {out}  organ_vol={organ_vol.shape} tumor_voxels={int(tumor.sum())}")


if __name__ == "__main__":
    main()
