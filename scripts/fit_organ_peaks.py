"""Regenerate src/tumor/assign.py::TYPICAL_PEAK from a validation run.

The competitive assigner divides each organ's similarity map by that organ's
typical peak response, so those constants are calibration data, not tuning
knobs. This script recomputes them from results/anatomy_validation/summary.json
so they can be refreshed whenever the checkpoint or patient set changes.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def fit_peaks(per_case):
    names = sorted(next(iter(per_case.values()))["peak_sim"])
    return {n: float(np.mean([v["peak_sim"][n] for v in per_case.values()]))
            for n in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/anatomy_validation/summary.json")
    a = ap.parse_args()
    s = json.loads(Path(a.summary).read_text())
    per_case = s["per_case"]
    peaks = fit_peaks(per_case)
    sd = {n: float(np.std([v["peak_sim"][n] for v in per_case.values()]))
          for n in peaks}
    print(f"# fitted on {len(per_case)} patients from {a.summary}")
    print("TYPICAL_PEAK = {")
    for n in sorted(peaks):
        print(f'    "{n}": {peaks[n]:.4f},'.ljust(34) + f"# sd {sd[n]:.4f}")
    print("}")


if __name__ == "__main__":
    main()
