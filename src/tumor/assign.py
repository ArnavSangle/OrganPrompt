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

BILATERAL PAIRS COMPETE AS ONE UNIT. Letting kidney-left and kidney-right bid
against each other reproduced the very bug this module exists to fix, one level
down: the two prompts differ by a single word ("LEFT side"/"RIGHT side"), the
text encoder separates them poorly, so both peak in nearly the same place and the
stronger one declared the weaker ABSENT. Measured over the 20 single-level
patients, 37% of pairs were split that way -- femur-right came back absent 9/20
against femur-left's 3/20, and one lumbar scan dropped lung-right while keeping
lung-left. A scan contains both sides of a pair or neither, so presence is
decided for the pair and the won territory is then divided by POSITION, which the
templates state explicitly and consistently: a patient-left organ "appears on the
RIGHT side of the image".
"""
import numpy as np
from scipy import ndimage

from src.labels.templates import CLASS_NAMES

# group key -> (image-left member, image-right member). Standard radiological
# orientation: the patient's LEFT organ appears on the image's RIGHT half, so
# "-left" organs sit at HIGH x. Verified against all three pairs of templates.
BILATERAL = {
    "femur": ("femur-right", "femur-left"),
    "kidney": ("kidney-right", "kidney-left"),
    "lung": ("lung-right", "lung-left"),
}

# organ name -> group key, and -> which image half it occupies (-1 low x, +1 high x)
GROUP_OF, SIDE = {}, {}
for _g, (_lo, _hi) in BILATERAL.items():
    GROUP_OF[_lo], SIDE[_lo] = _g, -1
    GROUP_OF[_hi], SIDE[_hi] = _g, +1
for _n in CLASS_NAMES.values():
    GROUP_OF.setdefault(_n, _n)          # unpaired organs are their own group

_GROUP_ID = {g: i for i, g in enumerate(sorted(set(GROUP_OF.values())))}

# Typical per-organ peak similarity: the mean of each organ's per-patient maximum,
# fitted on the 35 Spine-Mets patients that are NOT in the single-level evaluation
# set (regenerate with scripts/fit_organ_stats.py). Peak response is dominated by
# class identity, not by anatomy, so an UNCALIBRATED cross-organ argmax would hand
# nearly every voxel to the lungs. Dividing by these values puts each organ on a
# common "fraction of the response this organ ever gives" scale.
#
# These replace constants fitted by scripts/fit_organ_peaks.py on the evaluation
# set itself, which made every validation number circular. The correction is
# modest but systematic -- 10 of 11 organs rose, mean +7.0% -- and not uniform, so
# it reorders the argmax: pancreas +21.5% (3.8 sigma) moves from 9th to 6th, liver
# +12.1% from 4th to 3rd, bladder from 6th to 8th. A uniform shift would cancel in
# a ratio-based argmax; this one does not.
#
# Trailing sd is the spread across the 35 fitting patients, not a fit error.
TYPICAL_PEAK = {
    "bladder": 0.1196,       # sd 0.0327
    "femur-left": 0.0792,    # sd 0.0288
    "femur-right": 0.0628,   # sd 0.0239
    "heart": 0.1421,         # sd 0.0316
    "kidney-left": 0.1201,   # sd 0.0218
    "kidney-right": 0.1096,  # sd 0.0251
    "liver": 0.1584,         # sd 0.0361
    "lung-left": 0.1801,     # sd 0.0325
    "lung-right": 0.1810,    # sd 0.0285
    "pancreas": 0.1207,      # sd 0.0330
    "spleen": 0.1309,        # sd 0.0218
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
        # the winner is recorded as a GROUP, not an organ: bilateral pairs share
        # an id and therefore never take territory from each other
        self.best_grp = np.full(self.shape, unassigned, dtype=np.int8)
        self._peak = {}          # organ_class -> (calibrated value, coord)

    def _calibrate(self, organ_class, sim):
        name = CLASS_NAMES[organ_class]
        scale = self.peaks.get(name)
        if not scale:
            raise KeyError(f"no typical peak for organ {name!r}")
        return sim.astype(np.float32) / float(scale)

    def _gid(self, organ_class):
        return _GROUP_ID[GROUP_OF[CLASS_NAMES[organ_class]]]

    def _members(self):
        """group id -> [organ_class, ...], for the organs actually streamed."""
        out = {}
        for oc in self._peak:
            out.setdefault(self._gid(oc), []).append(oc)
        return out

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
        self.best_grp[win] = self._gid(organ_class)
        del cal, win

    def absent(self):
        """Organs in a group whose every member lost its own peak voxel.

        Decided per GROUP: one confident side vouches for its pair, because a
        scan that contains a left kidney contains a right one too.
        """
        out = set()
        for gid, members in self._members().items():
            if not any(self.best_grp[self._peak[oc][1]] == gid for oc in members):
                out.update(members)
        return out

    def _half(self, territory, organ_class):
        """The image half this organ occupies, per the templates' own convention."""
        mid = self.shape[-1] // 2
        out = np.zeros_like(territory)
        if SIDE[CLASS_NAMES[organ_class]] > 0:
            out[..., mid:] = territory[..., mid:]
        else:
            out[..., :mid] = territory[..., :mid]
        return out

    def _component(self, cand, coord):
        """The connected component of `cand` holding `coord`, else the largest.

        The fallback matters for bilateral organs: a member's own peak routinely
        lands in its pair's half (that poor left/right discrimination is the whole
        problem), so seeding from it would return nothing.
        """
        if not cand.any():
            return np.zeros(self.shape, dtype=bool)
        labeled, n = ndimage.label(cand)
        seed = int(labeled[coord]) if cand[coord] else 0
        if seed == 0:
            sizes = ndimage.sum(cand, labeled, range(1, n + 1))
            seed = int(np.argmax(sizes)) + 1
        return labeled == seed

    def finalize(self, floor=0.6):
        """Yield (organ_class, boolean mask), one organ at a time.

        A mask is the connected component of the group's won territory that
        contains the organ's own peak voxel; for a bilateral pair the territory is
        first cut on the image midline so each side gets its own half. `floor` is
        expressed as a fraction of the organ's typical peak: winning a voxel is not
        enough, the organ must also answer near the level it answers at when it
        really is present, or every voxel in the body would end up assigned to
        something.
        """
        gone = self.absent()
        for gid, members in self._members().items():
            if set(members) <= gone:
                for oc in members:
                    yield oc, np.zeros(self.shape, dtype=bool)
                continue
            territory = (self.best_grp == gid) & (self.best_val >= floor)
            # only split when both sides were actually streamed; a lone member
            # of a pair has nobody to share with and keeps the whole territory
            split = len(members) > 1
            for oc in members:
                cand = self._half(territory, oc) if split else territory
                yield oc, self._component(cand, self._peak[oc][1])
                if split:
                    del cand
            del territory
