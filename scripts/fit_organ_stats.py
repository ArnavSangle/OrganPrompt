"""Fit per-organ similarity statistics on patients held out from evaluation.

src/tumor/assign.py calibrates the cross-organ argmax by dividing each organ's
similarity map by a constant in TYPICAL_PEAK. Two things are wrong with those
constants, and this script produces the data to fix both.

LEAKAGE. TYPICAL_PEAK was fitted (scripts/fit_organ_peaks.py) on the summary of
scripts/validate_anatomy.py -- that is, on the 20 single-level Spine-Mets
patients that the anatomical validation then scores the method on. The
calibration the result depends on had already seen the test set. The other 35
patients on disk (26 multi-level, 9 with no lesions listed) carry no
single-level evaluation signal, so fitting on them and evaluating on the 20
breaks the circle. That split is the default here; --include_single_level puts
the leak back and exists only for measuring how much the leak was worth.

SCALE WITHOUT OFFSET. Dividing by a peak equalises each organ's range but not
its baseline. An organ whose response floor is high wins voxels on its floor
alone, which is the suspected mechanism behind the phantom detections that
survive competitive assignment. Correcting an offset needs each organ's mean and
standard deviation, so those are recorded alongside the peaks: peak_mean is a
drop-in replacement for today's TYPICAL_PEAK, and mean/sd are what a z-score
calibration would consume instead.

Nothing here modifies the calibration -- this script only measures.

Memory is the binding constraint. Full-resolution scans are streamed one organ
at a time via iter_organ_similarity and reduced to scalars immediately; the
stacked [11,D,H,W] array is never built, and no volume outlives the loop
iteration that produced it.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.spine_mets_labels import load_lesion_labels
from src.data.spine_mets import load_patient
from src.tumor.organ_volume import load_organcam, iter_organ_similarity
from src.labels.templates import CLASS_NAMES
from scripts.validate_anatomy import single_level_cases
import scripts.compute_index as ci


def fitting_cases(labels, on_disk, include_single_level=False):
    """Sorted case IDs to fit on: everything on disk minus the evaluation set.

    Driven by what is on disk rather than by the label table, because the 9
    patients with no lesions listed are legitimate scans for measuring organ
    response and would otherwise be dropped -- they appear in no lesion-level
    index and are the easiest quarter of the fitting set to lose by accident.
    """
    cases = set(on_disk)
    if not include_single_level:
        cases -= set(single_level_cases(labels))
    return sorted(cases)


def volume_stats(sim, chunk=16):
    """(peak, sum, sum_of_squares, count) for one organ volume, in float64.

    Reduced in slabs along z and accumulated in float64. Neither shortcut is
    safe: a float64 copy of a full-resolution volume is gigabytes, and a float32
    sum of squares loses precision fast over the ~10^8 voxels a single uncropped
    scan contributes.
    """
    a = np.asarray(sim)
    peak, total, total_sq, count = -np.inf, 0.0, 0.0, 0
    for i in range(0, a.shape[0], chunk):
        x = a[i:i + chunk].astype(np.float64)
        peak = max(peak, float(x.max()))
        total += float(x.sum())
        total_sq += float(np.square(x).sum())
        count += int(x.size)
        del x
    return peak, total, total_sq, count


class OrganStatsAccumulator:
    """Combine per-patient reductions into per-organ statistics.

    Holds scalars only -- five floats per organ plus one peak per patient -- so
    the whole 35-patient run costs a few kilobytes of state.
    """

    def __init__(self):
        self.peaks = {}          # organ -> [per-patient max, ...]
        self.sum = {}            # organ -> sum of every voxel, every patient
        self.sumsq = {}          # organ -> sum of squares, ditto
        self.count = {}          # organ -> voxel count, ditto
        self.per_case = {}       # case -> {organ: that patient's peak}

    def add_patient(self, case, stats):
        """Fold one patient's {organ: (peak, sum, sumsq, count)} into the totals.

        Called once per patient with every organ at once, so a patient that dies
        part-way through its organ stream contributes to none of them rather
        than to an arbitrary prefix.
        """
        self.per_case[case] = {}
        for name, (peak, total, total_sq, count) in stats.items():
            self.peaks.setdefault(name, []).append(float(peak))
            self.sum[name] = self.sum.get(name, 0.0) + float(total)
            self.sumsq[name] = self.sumsq.get(name, 0.0) + float(total_sq)
            self.count[name] = self.count.get(name, 0) + int(count)
            self.per_case[case][name] = float(peak)

    def result(self):
        """{organ: {peak_mean, peak_sd, mean, sd, n_patients, n_voxels}}.

        peak_mean uses TYPICAL_PEAK's own definition -- the mean of the
        per-patient maxima -- so it drops straight into the existing calibration.
        mean/sd are POOLED over every voxel of every patient, weighted by voxel
        count, which is what a z-score calibration needs; the mean of per-patient
        means or of per-patient sds would be a different (and wrong) quantity.
        """
        out = {}
        for name in sorted(self.peaks):
            peaks = self.peaks[name]
            n = self.count[name]
            mean = self.sum[name] / n if n else float("nan")
            # variance from the raw moments; clamped because catastrophic
            # cancellation can push a near-zero variance a hair below zero
            var = max(self.sumsq[name] / n - mean * mean, 0.0) if n else float("nan")
            out[name] = {
                "peak_mean": float(np.mean(peaks)),
                "peak_sd": float(np.std(peaks)),
                "mean": float(mean),
                "sd": float(np.sqrt(var)),
                "n_patients": len(peaks),
                "n_voxels": int(n),
            }
        return out


def patient_stats(ct, model, text_features, device, organs=None):
    """{organ_name: (peak, sum, sumsq, count)} for one patient.

    Streams one organ at a time and reduces each volume to four scalars before
    the next is generated -- the caller never holds two volumes at once.
    """
    out = {}
    for oc, sim in iter_organ_similarity(ct, model, text_features, device, organs):
        out[CLASS_NAMES[oc]] = volume_stats(sim)
        del sim
    return out


def run(root, checkpoint, device, out, stride=2, max_cases=0,
        include_single_level=False):
    labels = load_lesion_labels(str(Path(root) / "Spine-Mets-CT-SEG_Clinical.xlsx"))
    cases_dir = Path(root) / "spine_mets_ct_seg"
    on_disk = {c.name for c in cases_dir.iterdir() if c.is_dir()}
    cases = fitting_cases(labels, on_disk, include_single_level)
    if max_cases:
        cases = cases[:max_cases]
    held_out = sorted(set(single_level_cases(labels)) & on_disk)
    print(f"[fit_organ_stats] {len(cases)} fitting patients "
          f"({len(held_out)} single-level evaluation patients "
          f"{'INCLUDED -- results are leaked' if include_single_level else 'excluded'})",
          flush=True)
    print(f"[fit_organ_stats] cases: {', '.join(cases)}", flush=True)

    model, tf = load_organcam(checkpoint, device)
    acc = OrganStatsAccumulator()
    used, skipped = [], []
    for i, case in enumerate(cases):
        print(f"[fit_organ_stats] ({i + 1}/{len(cases)}) {case} ...", flush=True)
        try:
            p = load_patient(root, case, labels)
            # full uncropped volume: an organ's true peak can sit anywhere in the
            # scan, and a tumor-centered crop would bias every constant downward.
            # Sparse subsampling only, never a range crop -- organs span many
            # consecutive slices, so a stride of 2 barely moves the peak.
            ct = p.ct[::stride] if stride > 1 else p.ct
            stats = patient_stats(ct, model, tf, device)
            del ct, p
        except Exception as e:
            print(f"[fit_organ_stats] {case}: SKIP ({e})", flush=True)
            skipped.append(case)
            continue
        acc.add_patient(case, stats)
        used.append(case)
        print("[fit_organ_stats]   peaks: "
              + ", ".join(f"{n}={v[0]:.4f}" for n, v in sorted(stats.items())),
              flush=True)

    organs = acc.result()
    res = {
        "root": str(root),
        "checkpoint": str(checkpoint),
        "stride": stride,
        "include_single_level": bool(include_single_level),
        "n_patients": len(used),
        "cases": used,
        "skipped": skipped,
        "held_out_single_level": held_out,
        "organs": organs,
        "per_case": acc.per_case,
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))

    print(f"[fit_organ_stats] fitted on {len(used)} patients "
          f"({len(skipped)} skipped: {', '.join(skipped) or 'none'})", flush=True)
    print("[fit_organ_stats] organ".ljust(28)
          + "peak_mean  peak_sd     mean       sd", flush=True)
    for name, v in organs.items():
        print(f"[fit_organ_stats]   {name:<22}"
              f"{v['peak_mean']:>9.4f}{v['peak_sd']:>9.4f}"
              f"{v['mean']:>11.4f}{v['sd']:>9.4f}", flush=True)
    # paste-ready, so refreshing the constant is a copy rather than a retype
    print("[fit_organ_stats] unleaked drop-in replacement for "
          "src/tumor/assign.py::TYPICAL_PEAK:", flush=True)
    print("TYPICAL_PEAK = {", flush=True)
    for name, v in organs.items():
        print(f'    "{name}": {v["peak_mean"]:.4f},'.ljust(34)
              + f"# sd {v['peak_sd']:.4f}", flush=True)
    print("}", flush=True)
    print(f"[fit_organ_stats] wrote {out_path}", flush=True)
    return str(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Spine-Mets-CT-SEG_v1_2024")
    ap.add_argument("--checkpoint", default=ci.DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stride", type=int, default=2,
                     help="use every Nth axial slice; full volumes x 11 organs are "
                          "slow and organs span many slices, so a stride of 2 barely "
                          "moves the peak locations")
    ap.add_argument("--out", default="results/organ_stats_heldout.json")
    ap.add_argument("--max_cases", type=int, default=0, help="0 = all")
    ap.add_argument("--include_single_level", action="store_true",
                     help="ALSO fit on the 20 single-level patients that "
                          "scripts/validate_anatomy.py evaluates on. This "
                          "reintroduces the leak these statistics exist to remove; "
                          "use it only to measure how large that leak was.")
    a = ap.parse_args()
    run(a.root, a.checkpoint, a.device, a.out, a.stride, a.max_cases,
        a.include_single_level)


if __name__ == "__main__":
    main()
