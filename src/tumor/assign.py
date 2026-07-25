"""Competitive (multi-class) organ assignment.

OrganCAM produces one similarity map per organ prompt, and the index pipeline
originally binarized each map independently at a strict percentile. That has a
structural consequence: EVERY organ always receives a mask, whether or not the
organ appears in the scan at all. A bladder prompt run over a chest CT still has
a global maximum somewhere, so it claims the most bladder-like thoracic tissue
and reports itself as present. Absence is simply not expressible.

Measured on the 20 single-level Spine-Mets patients, this is not a rare edge
case: 5 of 20 patients top-ranked an anatomically impossible organ, three of
them the bladder. Neither component size nor peak similarity separates those
phantoms from correct detections -- a correct spleen came in at the 0th size
percentile while a wrong bladder sat at the 65th -- so no per-organ gate can
remove them.

Competition can. Assign each voxel to at most one organ: the organ that answers
most strongly there, after calibrating for the fact that raw similarity is not
comparable across prompts (lungs answer ~0.18 at their peak, femurs ~0.06). An
organ whose own best voxel is claimed by a stronger organ is reported ABSENT and
receives no mask -- which is exactly the signal independent thresholding cannot
produce.

Assignment streams one organ at a time, holding only a running best-value and
best-label array, so memory stays bounded regardless of organ count.
"""
import numpy as np
from scipy import ndimage

from src.labels.templates import CLASS_NAMES

# Typical per-organ peak similarity: the mean of each organ's per-patient maximum
# over the 20 single-level Spine-Mets patients (regenerate with
# scripts/fit_organ_peaks.py). Peak response is dominated by class identity, not
# by anatomy, so an UNCALIBRATED cross-organ argmax would hand nearly every voxel
# to the lungs. Dividing by these values puts each organ on a common "fraction of
# the response this organ ever gives" scale.
TYPICAL_PEAK = {
    "bladder": 0.1110,
    "femur-left": 0.0751,
    "femur-right": 0.0590,
    "heart": 0.1439,
    "kidney-left": 0.1103,
    "kidney-right": 0.1003,
    "liver": 0.1412,
    "lung-left": 0.1778,
    "lung-right": 0.1753,
    "pancreas": 0.0993,
    "spleen": 0.1281,
}


class CompetitiveAssigner:
    """Stream per-organ similarity volumes, emit mutually exclusive organ masks.

    Usage mirrors iter_organ_similarity:

        a = CompetitiveAssigner(ct.shape)
        for oc, sim in iter_organ_similarity(...):
            a.update(oc, sim)
        for oc, mask in a.finalize(floor=0.6):
            ...

    Memory is two volume-sized arrays (float32 + int8) no matter how many organs
    are streamed, plus one transient label array per organ during finalize.
    """

    def __init__(self, shape, peaks=None, unassigned=-1):
        self.shape = tuple(shape)
        self.peaks = dict(TYPICAL_PEAK if peaks is None else peaks)
        self.unassigned = unassigned
        self.best_val = np.full(self.shape, -np.inf, dtype=np.float32)
        self.best_lbl = np.full(self.shape, unassigned, dtype=np.int8)
        self._peak = {}          # organ_class -> (calibrated value, coord)

    def _calibrate(self, organ_class, sim):
        name = CLASS_NAMES[organ_class]
        scale = self.peaks.get(name)
        if not scale:
            raise KeyError(f"no typical peak for organ {name!r}")
        return sim.astype(np.float32) / float(scale)

    def update(self, organ_class, sim):
        """Fold one organ's similarity volume into the running argmax."""
        sim = np.asarray(sim)
        if sim.shape != self.shape:
            raise ValueError(f"expected shape {self.shape}, got {sim.shape}")
        cal = self._calibrate(organ_class, sim)
        coord = np.unravel_index(int(np.argmax(cal)), cal.shape)
        self._peak[organ_class] = (float(cal[coord]), coord)
        # strict > keeps assignment deterministic under ties: the first organ
        # streamed holds the voxel, and finalize's exclusivity check still holds
        win = cal > self.best_val
        self.best_val[win] = cal[win]
        self.best_lbl[win] = organ_class
        del cal, win

    def absent(self):
        """Organs whose own peak voxel was claimed by a stronger organ."""
        return {oc for oc, (_, coord) in self._peak.items()
                if self.best_lbl[coord] != oc}

    def finalize(self, floor=0.6):
        """Yield (organ_class, boolean mask), one organ at a time.

        A mask is the connected component of the organ's won territory that
        contains the organ's own peak voxel. `floor` is expressed as a fraction of
        the organ's typical peak: winning a voxel is not enough, the organ must
        also answer near the level it answers at when it really is present, or
        every voxel in the body would end up assigned to something.
        """
        for oc, (_, coord) in self._peak.items():
            if self.best_lbl[coord] != oc:
                yield oc, np.zeros(self.shape, dtype=bool)   # absent
                continue
            cand = (self.best_lbl == oc) & (self.best_val >= floor)
            if not cand[coord]:
                yield oc, np.zeros(self.shape, dtype=bool)   # peak below floor
                continue
            labeled, _ = ndimage.label(cand)
            seed = labeled[coord]
            mask = labeled == seed if seed else np.zeros(self.shape, dtype=bool)
            del cand, labeled
            yield oc, mask
