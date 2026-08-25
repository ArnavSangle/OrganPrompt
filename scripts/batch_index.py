"""Compute the bone-metastasis index for every labeled Spine-Mets patient."""
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
    ap.add_argument("--force", action="store_true",
                     help="recompute even if results/bone_index/<case>.json already exists")
    ap.add_argument("--percentile", type=float, default=99.9)
    ap.add_argument("--stride", type=int, default=1,
                     help="use every Nth axial slice; helps bound runtime for patients "
                          "with very wide tumor-bearing slice ranges")
    a = ap.parse_args()
    labels = load_lesion_labels(str(Path(a.root) / "Spine-Mets-CT-SEG_Clinical.xlsx"))
    cases_dir = Path(a.root) / "spine_mets_ct_seg"
    todo = [c for c in sorted(cases_dir.iterdir()) if c.is_dir() and labels.get(c.name)]
    print(f"[batch_index] {len(todo)} labeled patients to process", flush=True)
    per_case = {}
    for i, c in enumerate(todo):
        cached = Path(a.out_dir, f"{c.name}.json")
        if not a.force and cached.exists():
            per_case[c.name] = json.loads(cached.read_text())["index"]
            print(f"[batch_index] ({i + 1}/{len(todo)}) {c.name}: {per_case[c.name]:.3f} (cached)",
                  flush=True)
            continue
        print(f"[batch_index] ({i + 1}/{len(todo)}) {c.name} ...", flush=True)
        try:
            out = ci.run(a.root, c.name, a.checkpoint, a.device, a.alpha, a.beta,
                         a.limit_slices, a.out_dir, a.percentile, a.stride)
            per_case[c.name] = json.loads(Path(out).read_text())["index"]
            print(f"{c.name}: {per_case[c.name]:.3f}", flush=True)
        except Exception as e:
            print(f"{c.name}: SKIP ({e})", flush=True)
    summary = {"per_case": per_case, **aggregate_summary(per_case)}
    Path(a.out_dir, "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"mean index {summary['mean']:.3f} over {summary['n']} patients")


if __name__ == "__main__":
    main()
