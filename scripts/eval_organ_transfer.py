"""Organ-transfer sanity check: does OrganCAM (trained on TotalSegmentator) hold
up on CT-ORG's different acquisition protocol? Run manually before trusting the
bone-metastasis index."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.ct_org import load_ctorg_case
from src.tumor.organ_volume import load_organcam, build_organ_volume
from src.tumor.index import binarize_organs
from src.labels.templates import CLASS_NAMES

# CT-ORG label int -> OrganCAM class indices (liver, lungs L+R, kidneys L+R, bladder)
CTORG_TO_ORGANCAM = {1: [6], 3: [7, 8], 4: [4, 5], 2: [0]}
DEFAULT_CKPT = "checkpoints/totalseg_patch_v5/epoch_011_val3.4584.pt"


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum()) / float(u) if u else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stride", type=int, default=4)  # subsample slices for speed
    a = ap.parse_args()
    print(f"[eval_organ_transfer] loading {a.vol} ...", flush=True)
    ct, lab, _ = load_ctorg_case(a.vol, a.labels)
    ct, lab = ct[::a.stride], lab[::a.stride]
    print(f"[eval_organ_transfer] evaluating {ct.shape[0]} slices (stride={a.stride})", flush=True)
    model, tf = load_organcam(a.checkpoint, a.device)
    organs = sorted({o for lst in CTORG_TO_ORGANCAM.values() for o in lst})
    vol = build_organ_volume(ct, model, tf, a.device, organs)
    ob = binarize_organs(vol, {}, organs)   # threshold 0.0 (raw sims)
    results = {}
    for ctorg_lbl, oc_list in CTORG_TO_ORGANCAM.items():
        gt = lab == ctorg_lbl
        pred = np.zeros_like(gt)
        for oc in oc_list:
            pred |= ob[oc]
        name = "+".join(CLASS_NAMES[o] for o in oc_list)
        results[name] = iou(pred, gt)
        print(f"[eval_organ_transfer] {name}: IoU={results[name]:.4f}", flush=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
