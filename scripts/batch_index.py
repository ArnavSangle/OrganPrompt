"""Compute the bone-metastasis index for every labelled Spine-Mets patient."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.spine_mets_labels import load_lesion_labels
import scripts.compute_index as ci


def aggregate_summary(per_case):
    vals = np.array(list(per_case.values()), dtype=float)
    return {"n": int(vals.size), "mean": float(vals.mean()), "median": float(np.median(vals)),
            "min": float(vals.min()), "max": float(vals.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Spine-Mets-CT-SEG_v1_2024")
    ap.add_argument("--checkpoint", default=ci.DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--limit_slices", type=int, default=8)
    ap.add_argument("--out_dir", default="results/bone_index")
    a = ap.parse_args()
    labels = load_lesion_labels(str(Path(a.root) / "Spine-Mets-CT-SEG_Clinical.xlsx"))
    cases_dir = Path(a.root) / "spine_mets_ct_seg"
    per_case = {}
    for c in sorted(cases_dir.iterdir()):
        if not (c.is_dir() and labels.get(c.name)):
            continue
        try:
            out = ci.run(a.root, c.name, a.checkpoint, a.device, a.alpha, a.beta,
                         a.limit_slices, a.out_dir)
            per_case[c.name] = json.loads(Path(out).read_text())["index"]
            print(f"{c.name}: {per_case[c.name]:.3f}", flush=True)
        except Exception as e:
            print(f"{c.name}: SKIP ({e})", flush=True)
    summary = {"per_case": per_case, **aggregate_summary(per_case)}
    Path(a.out_dir, "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"mean index {summary['mean']:.3f} over {summary['n']} patients")


if __name__ == "__main__":
    main()
