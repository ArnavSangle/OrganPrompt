"""Qualitative overlay: tumor mask + top-scoring organ map on the peak tumor slice."""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
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
    ct = np.clip(p.ct[z], -150, 250)
    ct = (ct + 150) / 400
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(ct, cmap="gray")
    ax[0].imshow(np.ma.masked_where(~p.tumor_mask[z], p.tumor_mask[z]), cmap="autumn", alpha=0.7)
    ax[0].set_title(f"{a.case}: tumor (slice {z})")
    ax[0].axis("off")
    ax[1].imshow(ct, cmap="gray")
    ax[1].imshow(vol[top, 0], cmap="jet", alpha=0.5)
    ax[1].set_title(f"top organ map: {CLASS_NAMES[top]}")
    ax[1].axis("off")
    out = Path("figures/bone_index")
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{a.case}_overlay.png", dpi=110)
    print(f"wrote {out / (a.case + '_overlay.png')}")


if __name__ == "__main__":
    main()
