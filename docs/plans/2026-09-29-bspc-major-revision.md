# BSPC Major Revision (BSPC-D-26-16296) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer every comment from Reviewers 3 and 5 by fixing the three confirmed Phase 2 defects, rerunning the downstream validation once under a pre-registered configuration, revising the paper's claims, and writing a point-by-point response.

**Architecture:** Fix correctness first, because every Phase 2 number depends on it: SEG-to-CT geometry (Task 1), then calibration leakage (Task 2), then an honest definition of the index (Task 3). Freeze a pre-registration (Task 4), rerun once (Task 5), and only then rewrite prose and figures around whatever comes back (Tasks 6–12). The response letter comes last (Task 13).

**Tech Stack:** Python, pydicom, numpy, pytest, LaTeX (`elsarticle` via `scripts/build_bspc.py`).

**Spec:** Reviewer comments in Editorial Manager (Reviewers 3 and 5, Sept 16/18 2026); PDF export at `BSPC-D-26-16296-reviews.zip` (repo root). **Reviewer 5's text is truncated at source,** mid-sentence at the end of point 4 (the PDF export stops there too). The summary's four concerns map one-to-one onto points 1–4, so probably only point 4's closing request is missing. The editorial office has been asked for the full text (2026-09-29). Any further points get new tasks.

## Global Constraints

- Deadline: **Dec 01, 2026**. Target: all reruns done by **Oct 31**, leaving November for writing.
- Edit prose in `paper/main_short.tex` only. Regenerate `paper/main_bspc.tex` with `python scripts/build_bspc.py`. Never hand-edit `main_bspc.tex`.
- Pre-register every evaluation-set run: commit the config before running and report whatever comes back (CLAUDE.md standing rule).
- Do not tune ground truth to results. The geometry fix (Task 1) is decided from DICOM headers, not from the metric.
- Never push. Commit only when the user asks. No AI attribution in commit messages.
- Do not suggest higher-resolution input. The 14×14 grid is fixed.
- Phase 1 checkpoint `checkpoints/totalseg_patch_v5/epoch_011_val3.4584.pt` stays frozen. Only Task 7 (optional baseline) trains anything new.

## Review Focus

- A SEG whose `ImageOrientationPatient` matches the CT (no flip) must load unchanged. The fix must be geometric, not a hard-coded `[:, ::-1, :]`.
- A SEG with a different in-plane origin or FOV from the CT (offset IPP) must land in the right voxels, or fail loudly. Silent misplacement is what caused this bug.
- A patient in the cross-fit whose fold contains all of its own data must never see its own peaks. Test that fold membership and fit membership are disjoint for every patient.
- The 9 no-lesion patients must appear in every fold's fit set and in no fold's evaluation set.
- If the rerun moves the headline (e.g. p no longer < 0.0001, or the bladder/femur story changes), the paper reports the new numbers. Nothing gets reverted to protect the old narrative.

---

## Confirmed findings (verified 2026-09-29, before planning)

| Reviewer point | Status | Evidence |
|---|---|---|
| R5.3 image–mask misalignment | **Real bug, affects every Phase 2 number** | SEG `ImageOrientationPatient` = `(1,0,0, 0,-1,0)`, CT = `(1,0,0, 0,1,0)`, both cases checked. `load_vertebra_masks` (`src/data/spine_mets.py:64-91`) matches frames by z only and ignores in-plane geometry. Mean HU inside the vertebra masks as loaded: −198 to +65 (air or soft tissue). Row-flipped: 277–499 (bone). Every lesion was scored from its mirror image across the AP axis. |
| R5.2 calibration/eval overlap | **Real leak** | `TYPICAL_PEAK` was fitted on 35 = 26 multi-level + 9 no-lesion patients (`scripts/fit_organ_stats.py:8-14`). The per-lesion eval uses 46 = 20 single-level + those same 26. So 26/46 eval patients helped fit the calibration. Paper line 473 ("35 patients outside the evaluation set") is false for the 46-patient result. |
| R5.4 overlap term identically zero | **True** | `scripts/validate_lesions.py:94` does `mask &= ~tumor` before scoring, so `overlap` at lesion coords is always 0 and score = 0.5/(1+d). Ranking is by distance alone. The *ranks* are unaffected (monotone), but Eq. 2 misdescribes the method. |
| R5.1 "weakly supervised" | **Claim is wrong** | Training uses GT masks downsampled to 14×14 patch targets. That is coarse spatial supervision, not image-level. |

Orientation of the rest of the paper: CT-ORG (NIfTI) and TotalSegmentator (NIfTI) use different loaders and need their own check in Task 1, Step 6.

---

### Task 1: Geometry-correct SEG→CT mapping (R5.3)

**Files:**
- Modify: `src/data/spine_mets.py:64-91` (`load_vertebra_masks`)
- Test: `tests/test_spine_mets_volume.py`
- Create: `scripts/check_mask_alignment.py` (all-patient HU audit, writes `results/mask_alignment_audit.json`)

**Interfaces:**
- Produces: `load_vertebra_masks(seg_path, ct_paths) -> {level: bool[D,H,W]}` (same signature, now geometry-correct); `_resample_seg_frame(frame, seg_iop, seg_ipp, seg_ps, ct_iop, ct_ipp, ct_ps, ct_shape) -> bool[H,W]` (IPP as in-plane `(x, y)`, PixelSpacing as DICOM `(row, col)` spacing).

- [ ] **Step 1: Write the failing unit test (synthetic geometry, no data needed)**

```python
import numpy as np
import pytest
from src.data.spine_mets import _resample_seg_frame

def test_flipped_row_direction_is_unflipped():
    # CT: rows run +y from y0=-100. SEG: rows run -y from y0=+100 (opposite end).
    frame = np.zeros((4, 3), bool); frame[0, 1] = True     # SEG row 0 = y=+100 end
    out = _resample_seg_frame(
        frame,
        seg_iop=(1, 0, 0, 0, -1, 0), seg_ipp=(0.0, 100.0), seg_ps=(1.0, 1.0),
        ct_iop=(1, 0, 0, 0, 1, 0),  ct_ipp=(0.0, 97.0),   ct_ps=(1.0, 1.0),
        ct_shape=(4, 3))
    assert out[3, 1] and out.sum() == 1                   # lands at CT's last row

def test_matching_geometry_is_identity():
    frame = np.random.default_rng(0).random((5, 5)) > 0.5
    out = _resample_seg_frame(frame, (1,0,0,0,1,0), (0.,0.), (1.,1.),
                              (1,0,0,0,1,0), (0.,0.), (1.,1.), (5, 5))
    assert (out == frame).all()

def test_offset_origin_shifts_mask():
    frame = np.zeros((5, 5), bool); frame[0, 0] = True
    out = _resample_seg_frame(frame, (1,0,0,0,1,0), (2.,1.), (1.,1.),
                              (1,0,0,0,1,0), (0.,0.), (1.,1.), (5, 5))
    assert out[1, 2] and out.sum() == 1                    # x+2 cols, y+1 row

def test_non_axis_aligned_raises():
    with pytest.raises(ValueError):
        _resample_seg_frame(np.ones((2, 2), bool), (0.9,0.1,0,0,1,0), (0.,0.), (1.,1.),
                            (1,0,0,0,1,0), (0.,0.), (1.,1.), (2, 2))
```

- [ ] **Step 2: Run, verify it fails** with `ImportError: _resample_seg_frame`.
  Run: `pytest tests/test_spine_mets_volume.py -v`

- [ ] **Step 3: Implement.** Map every SEG pixel's patient-space (x, y) to CT (row, col) through both headers. Only axis-aligned orientations (cosines in {−1, 0, 1}) are supported; anything else raises, it doesn't guess.

```python
def _resample_seg_frame(frame, seg_iop, seg_ipp, seg_ps, ct_iop, ct_ipp, ct_ps, ct_shape):
    """Place one SEG frame on the CT's in-plane grid using both headers' geometry.

    DICOM: IOP = (row-direction cosines, column-direction cosines); the pixel at
    (r, c) sits at IPP + c*ps[1]*rowdir + r*ps[0]*coldir. Spine-Mets SEGs store
    column direction (0,-1,0) against the CT's (0,1,0), i.e. anterior-posterior
    flipped -- matching by z alone mirrored every lesion (see
    docs/plans/2026-09-29-bspc-major-revision.md).
    """
    def _axes(iop):
        v = np.asarray(iop, dtype=float).reshape(2, 3)
        if not np.allclose(np.abs(v), np.round(np.abs(v)), atol=1e-4):
            raise ValueError(f"non axis-aligned orientation {iop}")
        return v
    s, c = _axes(seg_iop), _axes(ct_iop)
    if not (np.allclose(np.abs(s[0]), np.abs(c[0])) and np.allclose(np.abs(s[1]), np.abs(c[1]))):
        raise ValueError(f"SEG/CT axes differ beyond sign: {seg_iop} vs {ct_iop}")
    r, k = np.nonzero(frame)
    # patient-space in-plane coordinates of each foreground SEG pixel (x, y only)
    pos = (np.asarray(seg_ipp[:2], float)[None]
           + k[:, None] * seg_ps[1] * s[0, :2][None]
           + r[:, None] * seg_ps[0] * s[1, :2][None])
    rel = pos - np.asarray(ct_ipp[:2], float)[None]
    ct_c = np.rint(rel @ c[0, :2] / ct_ps[1]).astype(int)
    ct_r = np.rint(rel @ c[1, :2] / ct_ps[0]).astype(int)
    out = np.zeros(ct_shape, bool)
    ok = (ct_r >= 0) & (ct_r < ct_shape[0]) & (ct_c >= 0) & (ct_c < ct_shape[1])
    if (~ok).any():
        raise ValueError(f"{int((~ok).sum())} SEG pixels fall outside the CT grid")
    out[ct_r[ok], ct_c[ok]] = True
    return out
```

  Then in `load_vertebra_masks`: read the CT's IOP/IPP/PixelSpacing from `ct_paths[0]`. Read the SEG IOP from `SharedFunctionalGroupsSequence[0].PlaneOrientationSequence` (fall back to per-frame if absent) and PixelSpacing from `PixelMeasuresSequence`, and take IPP per frame from `PlanePositionSequence`. Replace `masks[level][z] |= frames[f_idx].astype(bool)` with `masks[level][z] |= _resample_seg_frame(...)`.

- [ ] **Step 4: Run unit tests, verify pass.** `pytest tests/test_spine_mets_volume.py -v`

- [ ] **Step 5: Add the real-data regression test** (skipped when data absent). Mean HU inside each vertebra mask must look like bone:

```python
@pytest.mark.skipif(not ROOT.exists(), reason="Spine-Mets not present")
@pytest.mark.parametrize("case", ["10355", "13683"])
def test_vertebra_masks_sit_on_bone(case):
    ct, seg = find_series(str(ROOT / case))
    vol, _ = load_ct_volume(ct)
    for lvl, m in load_vertebra_masks(seg, ct).items():
        assert vol[m].mean() > 150, f"{case}:{lvl} mean HU {vol[m].mean():.0f}"
```

  Pre-fix values for reference: 10355 T1 −113, T2 −136, T3 +65 (fail). Row-flipped: 302 / 414 / 499.

- [ ] **Step 6: All-patient audit.** `scripts/check_mask_alignment.py` loops all 55 cases and records per level the mean HU as loaded, plus both SEG and CT IOP. It writes `results/mask_alignment_audit.json` and prints any level below 150 HU. Expected: every level passes. Investigate any failure by hand (e.g. a collapsed or fractured vertebra) rather than lowering the threshold. Also: open one CT-ORG case with `scripts/debug_ctorg_overlay.py` and one TotalSegmentator case and confirm visually that label masks sit on the right anatomy. Record the outcome in the JSON's `other_datasets` field. That step answers the reviewer's "affine transformations" request.

- [ ] **Step 7: Confirm nothing else consumes in-plane lesion coords with its own convention.** `grep -rn "lesion_coords\|tumor_mask\|load_vertebra_masks" src scripts`. `scripts/scan_extent.py` uses z only and is unaffected. Note that in the audit.

- [ ] **Step 8: Commit** (when asked): `fix(spine_mets): map SEG frames onto CT grid via DICOM geometry`

---

### Task 2: Cross-fitted calibration (R5.2)

**Files:**
- Modify: `scripts/fit_organ_stats.py` (run once with `--include_single_level` so all 55 patients have `per_case` peaks, output `results/organ_stats_all55.json`)
- Create: `src/tumor/crossfit.py`
- Modify: `scripts/validate_lesions.py` (`score_patient` accepts `peaks=`, passes it to `CompetitiveAssigner(ct.shape, peaks=peaks, calibration=...)`; new `--crossfit results/crossfit_folds.json` flag)
- Create: `results/crossfit_folds.json` (committed **before** Task 5 runs)
- Test: `tests/test_crossfit.py`

**Interfaces:**
- Consumes: `CompetitiveAssigner(shape, peaks=None, ...)` (`src/tumor/assign.py:144`), `per_case` from the stats JSON: `{case: {organ_name: peak}}`.
- Produces: `make_folds(eval_cases: list[str], k: int, seed: int) -> dict[str, int]`; `fold_peaks(per_case: dict, folds: dict[str,int], always_fit: list[str], fold: int) -> dict[int, float]` (keyed by class int like `TYPICAL_PEAK`).

The per-patient organ peaks don't depend on lesion masks, so Task 1 doesn't invalidate them. The 35 existing `per_case` entries can be reused, and only the 20 single-level patients need a new inference pass.

- [ ] **Step 1: Failing tests**

```python
from src.labels.templates import CLASS_NAMES
from src.tumor.crossfit import make_folds, fold_peaks

EVAL = [f"c{i:02d}" for i in range(46)]
NOLES = [f"n{i}" for i in range(9)]

def test_folds_cover_eval_once_and_are_balanced():
    f = make_folds(EVAL, k=5, seed=0)
    assert set(f) == set(EVAL) and set(f.values()) == set(range(5))
    sizes = [list(f.values()).count(i) for i in range(5)]
    assert max(sizes) - min(sizes) <= 1

def test_folds_deterministic():
    assert make_folds(EVAL, 5, 0) == make_folds(EVAL, 5, 0)

def test_fold_peaks_excludes_own_fold_and_includes_no_lesion():
    per_case = {c: {"liver": 1.0} for c in EVAL} | {c: {"liver": 1.0} for c in NOLES}
    f = make_folds(EVAL, 5, 0)
    for c in EVAL:                       # poison each patient's own fold
        if f[c] == 2: per_case[c]["liver"] = 999.0
    p = fold_peaks(per_case, f, NOLES, fold=2)
    liver_id = next(k for k, v in CLASS_NAMES.items() if v == "liver")
    assert p[liver_id] == 1.0            # poisoned fold never seen
```

- [ ] **Step 2: Run, verify fail.** `pytest tests/test_crossfit.py -v`

- [ ] **Step 3: Implement**

```python
"""Patient-level cross-fitting for the calibration constants (TYPICAL_PEAK).

Every evaluation patient is scored with constants fitted on patients outside its
own fold, plus the 9 no-lesion patients (which are never evaluated and so sit in
every fold's fitting set). Replaces the single 35-patient fit, which overlapped
26 of the 46 per-lesion evaluation patients.
"""
import random
import numpy as np
from src.labels.templates import CLASS_NAMES

_ID = {v: k for k, v in CLASS_NAMES.items()}

def make_folds(eval_cases, k, seed):
    cases = sorted(eval_cases)
    random.Random(seed).shuffle(cases)
    return {c: i % k for i, c in enumerate(cases)}

def fold_peaks(per_case, folds, always_fit, fold):
    fit = [c for c, f in folds.items() if f != fold] + list(always_fit)
    organs = per_case[fit[0]].keys()
    return {_ID[o]: float(np.mean([per_case[c][o] for c in fit])) for o in organs}
```

- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Get per-case peaks for the 20 single-level patients:** `python scripts/fit_organ_stats.py --include_single_level --out results/organ_stats_all55.json` (same `--stride 2` and checkpoint as `organ_stats_heldout.json`). Check that the 35 overlapping `per_case` values match the old file to 1e-6. This is the determinism check.
- [ ] **Step 6: Wire `--crossfit`** into `validate_lesions.py`: load folds and the stats, and for each patient call `fold_peaks(..., fold=folds[case])` and pass `peaks=` through `score_patient`. Unit-test with a fake assigner that records the `peaks` it received.
- [ ] **Step 7: Generate and commit `results/crossfit_folds.json`** with `k=5, seed=20260929`, the 46 eval cases and the 9 no-lesion cases listed explicitly. This file is part of the pre-registration (Task 4).
- [ ] **Step 8: Commit.**

---

### Task 3: Make the index definition match the code (R5.4)

**Files:**
- Modify: `paper/main_short.tex` §3 index subsection (lines ~404–430, Eq. 1–2)
- Test: `tests/test_lesion_scoring.py`

**Decision (recommended):** Drop the overlap term from the paper and define the ranking as minimum surface distance d_mm from the involved vertebra to each predicted organ, ranked ascending. The code doesn't need to change: `score = 0.5/(1+d)` ranks identically. Say explicitly that vertebra voxels are removed from organ masks, which makes overlap undefined for this use. Rename "tumor mask" to **"involved-vertebra mask"** throughout, and state as a limitation that Spine-Mets gives vertebra-level involvement, not lesion extent.

- [ ] **Step 1:** Add a test that pins the equivalence, so the text can't drift from the code again:

```python
def test_rank_is_distance_rank_after_carveout():
    # after mask &= ~tumor, overlap==0 and score ordering == distance ordering
    recs = {"a": {"overlap": 0.0, "distance_mm": 3.0}, "b": {"overlap": 0.0, "distance_mm": 10.0}}
    scores = {k: 0.5 / (1 + v["distance_mm"]) for k, v in recs.items()}
    assert sorted(scores, key=scores.get, reverse=True) == sorted(recs, key=lambda k: recs[k]["distance_mm"])
```

  Also add an integration assertion to `score_patient`'s existing test that every returned `overlap == 0.0`.
- [ ] **Step 2:** Rewrite the methods paragraph and the equations. Leave the numbers as placeholders (`\todo{}`) until Task 5.
- [ ] **Step 3:** Read the rest of Reviewer 5's comments (point 4 onward) and add tasks for anything new.

---

### Task 4: Pre-registration for the single rerun

**Files:** Create `docs/preregistration/2026-10-revision-rerun.md`, committed before Task 5.

- [ ] **Step 1:** Write down, before running anything:
  - Config: `--calibration scale --floor 0.6`, stride as in the published run, checkpoint v5, no negatives, `--crossfit results/crossfit_folds.json` (k=5, seed 20260929), geometry-fixed loader (Task 1 commit hash).
  - Primary endpoint: mean rank vs chance with patient-clustered bootstrap CI and the existing permutation p-value (`scripts/analyze_lesions.py`), plus top-1 and top-3 over all 180.
  - Secondary endpoints, reported descriptively: per-level table, bladder/femur top-rank counts, impossible top-1 count.
  - Variant reruns: rerun the two **already-reported** variants (global negatives, z-score k=2) with the fixed loader and cross-fit, same configs as before. This is a correction, not a new comparison. Their paper numbers are currently wrong too.
  - Subgroup analysis (Task 9), declared as exploratory only.
  - Commitment: the paper reports whatever comes back, including if the headline weakens.
- [ ] **Step 2:** Commit.

---

### Task 5: Run the corrected validation

**Files:** outputs `results/lesion_validation_v2/`, `results/lesion_validation_negatives_v2/`, `results/lesion_validation_zscore_v2/`

- [ ] **Step 1:** Check VRAM (`nvidia-smi`) and run the three configurations one after another, not in parallel. Verify each process directly (`Get-Process`, command line) per the standing rule.
- [ ] **Step 2:** Run `scripts/analyze_lesions.py` on each output. Record the primary endpoint in the pre-registration doc as a dated "Results" addendum.
- [ ] **Step 3:** Rerun `scripts/validate_anatomy.py` (20-patient per-patient numbers) only if the paper still reports them. Otherwise remove them from the paper.
- [ ] **Step 4:** Regenerate all Phase 2 figures: `scripts/make_lesion_figures.py`, `scripts/make_qualitative_figure.py` (Figure 5), and the pipeline figure node images that show case 10355 (`scripts/make_pipeline_node_images.py`). Check that the red masks now sit on vertebrae.
- [ ] **Step 5:** Update CLAUDE.md "Current state" with the new numbers and mark the old ones as superseded by the geometry fix.
- [ ] **Step 6:** Stop and review with the user before writing prose. The Discussion's "two broken classes" framing may no longer hold.

---

### Task 6: Supervision reframing (R5.1)

**Files:** `paper/main_short.tex` (title line 37, abstract, intro, related work lines ~200–225, conclusion), `scripts/build_bspc.py` (title/highlights if hard-coded), `paper/highlights.txt`

- [ ] **Step 1:** Replace "weakly supervised" / "image-level tier" with an accurate description: training uses **coarse 14×14 patch-level targets derived from dense masks**, and inference uses text prompts only with no masks. Add a small table: *annotation needed at training* vs *at inference*, for OrganPrompt and for the related-work tiers.
- [ ] **Step 2:** New title candidate (user decides): *"OrganPrompt: Text-Prompted Organ Localization in CT from Coarse Patch-Level Supervision, with Application to Vertebral Metastasis–Organ Proximity"*. Changing the title in a revision is allowed. Mention it in the response.
- [ ] **Step 3:** Remove or qualify every sentence contrasting against "fully supervised" as if OrganPrompt used no spatial labels (e.g. the table at line ~542, "Dice (2D, weakly sup.)").

### Task 7 (decision needed): Equal-supervision baseline (R5.1)

Reviewer 5 asks to "compare against methods with equivalent supervision." Options, cheapest first:
- **(a) Linear probe on frozen BioMedCLIP patch tokens**, trained on the same 14×14 targets and split, evaluated with the same pipeline. About 1 day. It isolates what the text-alignment and contrastive fine-tuning add over "same features, same labels".
- **(b) Small CNN or U-Net trained on 14×14-downsampled masks**, upsampled identically. 2–4 days.
- **(c) Neither.** Argue in the response that the contribution is the text interface, not mIoU. This is weaker, and Reviewer 5 will probably insist.

Recommend (a). Pre-register it on the TotalSeg test split (a separate set from the Spine-Mets eval set). If chosen, it gets its own sub-plan with `scripts/train_linear_probe.py` + tests.

---

### Task 8: Phase 1 reporting fixes (R3.3, R3.4, R3.5, R3.6)

**Files:** `paper/main_short.tex`, `src/data/totalsegmentator.py:120-160` (read only), `scripts/make_qualitative_figure.py` or a new `scripts/make_organ_slice_grid.py`

- [ ] **R3.3 split:** Print volume and slice counts per split from `src/data/totalsegmentator.py` (seed 42) and add them as a table. Extract `meta.csv` from `data/TotalSeg/Totalsegmentator_dataset_v201.zip`, check whether any patient identifier exists, and state plainly what leakage can and can't be ruled out. Don't claim patient-level separation unless the metadata supports it. Current text line 348 says "no slice from a training patient", which conflates volume and patient.
- [ ] **R3.4 Table 2:** Load each `checkpoints/totalseg_patch{,_v2,_v3,_v4,_v5}` checkpoint's saved args/optimizer state (and `checkpoint_metrics.json` where present) plus `git log -p scripts/train.py` to record LR, epochs, batch size, loss weights and data per stage. Add columns (epochs, LR) to Table 2. If Patch-tuned → II is "more epochs at the same settings", say so and call it continued training, not a method change.
- [ ] **R3.5 Figure 2 selection:** State the selection rule used. Add a supplementary figure with a fixed rule: per organ, test-split slices at the 25th/50th/75th percentile of that organ's z-extent in 3 test volumes chosen by seed. Deterministic, not cherry-picked.
- [ ] **R3.6 L5 adversarial:** From the repo prompt sets (`scripts/caption_ablation.py`, `scripts/make_prompt_appendix.py`), list the exact swapped tokens. Add the motivation (tests whether the model reads position language or ignores it) and why LEFT/RIGHT and top/bottom were chosen (they're the only position axes the templates encode). Name untested alternatives (organ-name swap, negation, shuffled sentence order) as limitations. Text only.

### Task 9: Exploratory subgroups (R3.7)

**Files:** Create `scripts/subgroup_lesions.py`, test `tests/test_subgroup_lesions.py`

- [ ] Join `results/lesion_validation_v2/` per-lesion ranks with `Spine-Mets-CT-SEG_Clinical.xlsx` (Age, Sex, Primary cancer) and lesion type (lytic/blastic/mixed, already in the loader). Report mean rank and top-1 per subgroup: sex; age split at the cohort median; primary cancer with at least 5 patients, others pooled; lesion type. Include per-subgroup chance and n patients / n lesions. **No per-subgroup p-values.** Label the table exploratory, and state that n=46 can't support subgroup inference, which answers the reviewer honestly.
- [ ] Test: the join drops no lesions, and a subgroup's lesion counts sum to 180.

### Task 10: Cohort flow diagram and split lists (R5.2)

- [ ] Flow diagram (TikZ in the paper or a matplotlib script): 55 → 9 no lesions (calibration-only) → 46 evaluated (20 single-level + 26 multi-level) → 180 lesions, with the 5 cross-fit folds. Ship `supplementary/spine_mets_splits.csv` (TCIA case IDs are already public and de-identified) with columns `case, role, fold`.
- [ ] Rewrite the "exposure" paragraph (line ~466): say that pipeline development touched the eval set. Present the downstream result as **exploratory**, as the reviewer asks, unless an independent cohort exists. (Check whether any public dataset with vertebral-level metastasis labels could serve. If not, say so.)

### Task 11: Text-only fixes (R3.1, R3.2, R3.8, R3.9)

- [ ] **R3.1:** Rephrase "frozen in the label set supported by a funding decision" with a cited, factual statement about annotation cost and fixed label sets. Verify every citation exists (DOI resolves) before adding it.
- [ ] **R3.2:** Fetch and read the four suggested papers (10.1007/s10462-025-11151-8, 10.1007/s10462-025-11193-y, 10.1016/j.engappai.2025.110794, 10.1016/j.engappai.2026.115333). Cite the ones that are actually relevant in Related Work. Don't run them. For any that aren't relevant, say so politely in the response. Reviewers sometimes suggest their own work, and editors accept a reasoned decline.
- [ ] **R3.8:** Add a paragraph on 3D extension: the current pipeline is 2D per slice plus 3D connected components. Name concrete options: a 2.5D input feeding adjacent slices as the three input channels (fits the ViT's existing 3-channel input with no resolution change), z-smoothing of similarity volumes before assignment, and 3D CRF. Note the inter-slice consistency risk seen in the phantom analysis.
- [ ] **R3.9:** Normalize `paper/references.bib` arXiv entries (`eprint`/`archivePrefix`, one DOI style `https://doi.org/...`). Rebuild and check refs [12], [13] and every other arXiv entry in the PDF.

### Task 12: Rebuild the submission

- [ ] `python scripts/build_bspc.py`, then `cd paper && latexmk -pdf main_bspc.tex`. Zero undefined refs/citations in the log.
- [ ] Produce a revised manuscript with changes marked (latexdiff old vs new `main_bspc.tex`) plus a clean version, and a source zip (tex, cls/bst, bib or bbl, figures) per the email.

### Task 13: Response to reviewers

**Files:** Create `paper/revision/response_to_reviewers.tex` (or .docx if preferred)

- [ ] One entry per comment: quote → response → change made with page/line in the marked-up PDF. Every comment in the Editorial Manager review platform also needs a reply there ("You must reply to every comment").
- [ ] Lead the R5.3 reply by thanking the reviewer and stating plainly: the SEG/CT orientation mismatch was real, how it was found and fixed (geometry-based, with a bone-HU audit on all 55 patients), and old vs new headline numbers side by side. The same goes for R5.2 (leak) and R5.4 (overlap term). Owning these directly is what makes a major revision succeed.
- [ ] Summary of changes at the top for the editor.

---

## Suggested schedule

| Week of | Work |
|---|---|
| Sep 29 | Tasks 1–2 (bugs), paste remaining R5 comments |
| Oct 6 | Tasks 3–4, decide Task 7, run Task 5 |
| Oct 13 | Review rerun results with user; Task 7 if chosen |
| Oct 20–Nov 9 | Tasks 6, 8–11 (writing) |
| Nov 10–23 | Tasks 12–13, advisor read-through |
| Nov 24 | Submit (one week of buffer before Dec 1) |

---

## Final figures pass: reminders for the user (added 2026-09-29)

**Figure 1 (pipeline) is assembled externally; the user updates it by hand.**
Node images come from `scripts/make_pipeline_node_images.py` → `figures/paper/pipeline_nodes/`.

| Node | Content | Status after the revision |
|---|---|---|
| node1_ct_slice, node5_gt_mask_pooled | TotalSegmentator slice, pooled GT mask | Unaffected |
| node8_similarity_map, node9_organ_mask | 10355 T3, lung-left prompt, raw similarity | Not affected by the mirroring (no vertebra drawn; z was always right). But the revised run ranks **lung-right** first for 10355 T3 — check any "top organ" wording in Figure 1 |
| **node11_metastasis_index** | 10355 patient-level "index" with the tumor mask overlaid | **Must be redone**: overlay used the mirrored vertebra mask, and the patient-level blended "index" no longer exists (now a distance ranking from the involved vertebra). Rename the box, e.g. "Organ ranking by distance from involved vertebra" |
| Panel B caption / box labels | "Metastasis index", "tumor" | Rename to involved vertebra / distance ranking to match the text |

`make_pipeline_node_images.py` imports `render_case, CASES` from `make_qualitative_figure.py`, which the revision changed — fix the import before rerunning it.

**Replacement for node 11:** `python scripts/make_node11_ranking.py` → `figures/paper/pipeline_nodes/node11_distance_ranking.png` (10355 T3, assigned lung-right mask, vertebra outline, nearest-pair segment, label "lung-right, 27 mm"). In the external Figure 1: swap the image, relabel "Metastasis Index — α·overlap + β·proximity" → "Organ ranking by distance from involved vertebra", relabel "overlap + KD-tree distance" → "KD-tree distance", drop "index = 0.202". Nodes 8/9 show the lung-LEFT prompt while the ranking now puts lung-RIGHT first — either note it in the caption or regenerate them with the lung-right prompt.

**Figures regenerated in this revision** (all in `paper/figures_for_upload/`): `alignment_multiplanar.png` (new), `detection_vs_phantom.png` (Figure 5, assigned masks, 3 cases), `per_level_rank.png`, `rank_distribution.png` (replaces `index_distribution.png`), `organ_extent_S1.png` (supplementary, new). The flow diagram is TikZ inside the .tex.
