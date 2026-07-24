"""Ad hoc diagnostic: visualize GT vs predicted organ mask on one CT-ORG slice.

Not part of the Phase 1 pipeline — a one-off tool to inspect *why* organ-transfer
IoU is low/zero for a given organ (orientation mismatch vs. noisy false positives
vs. genuinely missed detection look very different on an actual image).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.ct_org import load_ctorg_case
from src.tumor.organ_volume import load_organcam, build_organ_volume
from src.labels.templates import CLASS_NAMES

DEFAULT_CKPT = "checkpoints/totalseg_patch_v5/epoch_011_val3.4584.pt"
CTORG_LABEL_TO_ORGANCAM = {1: [6], 3: [7, 8], 4: [4, 5], 2: [0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ctorg_label", type=int, required=True,
                     help="CT-ORG GT label to inspect (1 liver, 2 bladder, 3 lungs, 4 kidneys)")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--slice", type=int, default=None,
                     help="Slice index to inspect (default: the one with the most GT voxels)")
    a = ap.parse_args()

    print(f"[debug_ctorg_overlay] loading {a.vol} ...", flush=True)
    ct, lab, _ = load_ctorg_case(a.vol, a.labels)
    gt = lab == a.ctorg_label
    z = a.slice if a.slice is not None else int(np.argmax(gt.sum(axis=(1, 2))))
    print(f"[debug_ctorg_overlay] inspecting slice {z} (gt voxels here: {int(gt[z].sum())})",
          flush=True)

    organs = CTORG_LABEL_TO_ORGANCAM[a.ctorg_label]
    model, tf = load_organcam(a.checkpoint, a.device)
    vol = build_organ_volume(ct[z:z + 1], model, tf, a.device, organs)  # (n_organs,1,H,W)
    pred = np.zeros(ct.shape[1:], dtype=bool)
    for oi, oc in enumerate(organs):
        pred |= vol[oi, 0] >= 0.0
        print(f"[debug_ctorg_overlay] {CLASS_NAMES[oc]} raw sim: "
              f"min={vol[oi, 0].min():.4f} max={vol[oi, 0].max():.4f} "
              f"mean={vol[oi, 0].mean():.4f}", flush=True)

    ct_disp = np.clip(ct[z], -150, 250)
    ct_disp = (ct_disp + 150) / 400

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(ct_disp, cmap="gray")
    ax[0].set_title(f"CT slice {z}")
    ax[0].axis("off")
    ax[1].imshow(ct_disp, cmap="gray")
    ax[1].imshow(np.ma.masked_where(~gt[z], gt[z]), cmap="Greens", alpha=0.6)
    ax[1].set_title(f"GT label {a.ctorg_label}")
    ax[1].axis("off")
    ax[2].imshow(ct_disp, cmap="gray")
    ax[2].imshow(np.ma.masked_where(~pred, pred), cmap="Reds", alpha=0.6)
    ax[2].set_title("+".join(CLASS_NAMES[o] for o in organs) + " prediction (raw sim >= 0)")
    ax[2].axis("off")

    out = Path("figures/debug")
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"ctorg_label{a.ctorg_label}_slice{z}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"[debug_ctorg_overlay] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
