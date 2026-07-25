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
from src.tumor.index import select_peak_component
from src.labels.templates import CLASS_NAMES

# CT-ORG label int -> OrganCAM class indices (liver, lungs L+R, kidneys L+R, bladder)
CTORG_TO_ORGANCAM = {1: [6], 3: [7, 8], 4: [4, 5], 2: [0]}
DEFAULT_CKPT = "checkpoints/totalseg_patch_v5/epoch_011_val3.4584.pt"


def scores(pred, gt):
    """IoU plus precision/recall.

    IoU alone is misleading here: select_peak_component keeps only the single
    connected component around each organ's peak voxel, so recall is capped by
    construction and IoU is dominated by how much of the organ the component
    happens to cover. The index only ever uses the component's *location*, so
    precision (fraction of the predicted component that lands inside the true
    organ) is the number that actually says whether transfer works.
    """
    inter = float((pred & gt).sum())
    union = float((pred | gt).sum())
    return {
        "iou": inter / union if union else float("nan"),
        "precision": inter / float(pred.sum()) if pred.any() else float("nan"),
        "recall": inter / float(gt.sum()) if gt.any() else float("nan"),
        "gt_voxels": int(gt.sum()),
        "pred_voxels": int(pred.sum()),
        "overlap_voxels": int(inter),
    }


def eval_case(vol_path, label_path, model, tf, device, stride, percentiles):
    print(f"[eval_organ_transfer] loading {vol_path} ...", flush=True)
    ct, lab, _ = load_ctorg_case(vol_path, label_path)
    print(f"[eval_organ_transfer] full volume: {ct.shape[0]} slices", flush=True)
    for lbl in sorted(set(CTORG_TO_ORGANCAM) | {0}):
        print(f"[eval_organ_transfer]   GT label {lbl}: {int((lab == lbl).sum())} voxels "
              f"across {int((lab == lbl).any(axis=(1, 2)).sum())} slices", flush=True)
    ct, lab = ct[::stride], lab[::stride]
    print(f"[eval_organ_transfer] evaluating {ct.shape[0]} slices (stride={stride})", flush=True)
    organs = sorted({o for lst in CTORG_TO_ORGANCAM.values() for o in lst})
    vol = build_organ_volume(ct, model, tf, device, organs)
    out = {}
    for pct in percentiles:
        ob = {oc: select_peak_component(vol[oi], pct) for oi, oc in enumerate(organs)}
        per_pct = {}
        for ctorg_lbl, oc_list in CTORG_TO_ORGANCAM.items():
            gt = lab == ctorg_lbl
            pred = np.zeros_like(gt)
            for oc in oc_list:
                pred |= ob[oc]
            name = "+".join(CLASS_NAMES[o] for o in oc_list)
            s = scores(pred, gt)
            per_pct[name] = s
            print(f"[eval_organ_transfer] p{pct} {name}: IoU={s['iou']:.4f} "
                  f"prec={s['precision']:.4f} rec={s['recall']:.4f} "
                  f"(gt={s['gt_voxels']}, pred={s['pred_voxels']}, "
                  f"overlap={s['overlap_voxels']})", flush=True)
        out[str(pct)] = per_pct
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", required=True, nargs="+")
    ap.add_argument("--labels", required=True, nargs="+")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stride", type=int, default=4)  # subsample slices for speed
    ap.add_argument("--percentile", type=float, nargs="+", default=[99.9, 99.0],
                     help="per-organ binarization percentiles to sweep")
    ap.add_argument("--out", default="results/organ_transfer.json")
    a = ap.parse_args()
    if len(a.vol) != len(a.labels):
        ap.error(f"--vol has {len(a.vol)} paths but --labels has {len(a.labels)}")
    model, tf = load_organcam(a.checkpoint, a.device)
    results = {}
    for i, (v, l) in enumerate(zip(a.vol, a.labels)):
        print(f"[eval_organ_transfer] === case {i + 1}/{len(a.vol)} ===", flush=True)
        results[Path(v).stem] = eval_case(v, l, model, tf, a.device, a.stride, a.percentile)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"[eval_organ_transfer] wrote {a.out}", flush=True)
    # mean precision per organ per percentile, across cases
    for pct in a.percentile:
        for name in next(iter(results.values()))[str(pct)]:
            ps = [results[c][str(pct)][name]["precision"] for c in results]
            ps = [p for p in ps if p == p]
            if ps:
                print(f"[eval_organ_transfer] MEAN p{pct} {name}: "
                      f"precision={sum(ps) / len(ps):.4f} over {len(ps)} cases", flush=True)


if __name__ == "__main__":
    main()
