# Pre-registration: BSPC revision rerun of the per-lesion validation

Written 2026-09-29, **before** any result produced with the corrected loader or
cross-fitted calibration has been seen. Committed before the runs start. The
paper reports whatever these runs return.

## Why a rerun

Three defects found while answering the BSPC reviews (BSPC-D-26-16296):

1. **Mirrored lesion masks (Reviewer 5, pt 3).** `load_vertebra_masks` pasted SEG
   frames onto the CT grid by z only. Spine-Mets SEGs store the column direction
   as (0,−1,0) against the CT's (0,1,0), so every involved-vertebra mask sat
   mirrored anterior–posterior (mean HU −198…+65 as loaded vs 277…499 flipped,
   cases 10355/13683). Fixed by mapping through DICOM IOP/IPP/PixelSpacing.
2. **Calibration overlap (Reviewer 5, pt 2).** TYPICAL_PEAK / ORGAN_MEAN / ORGAN_SD
   were fitted on 35 patients, 26 of whom are among the 46 evaluated. Replaced by
   5-fold patient-level cross-fitting.
3. **Unrecorded negatives.** Since commit d26258e `validate_lesions.py` applied
   NEGATIVE_PROMPTS on every run with no flag. Now opt-in; every run records its
   full configuration in `summary.json`.

## Fixed configuration

| Setting | Value |
|---|---|
| Checkpoint | `checkpoints/totalseg_patch_v5/epoch_011_val3.4584.pt` (frozen) |
| Loader | geometry-corrected `load_vertebra_masks` (`seg_geometry: dicom-iop-ipp`) |
| Stride | 2 |
| Calibration | cross-fitted, `results/crossfit_folds.json` (k=5, seed 20260929; 46 evaluated patients, 9 no-lesion patients in every fold's fit) |
| Organ statistics | `results/organ_stats_all55.json` (`fit_organ_stats.py --include_single_level`, stride 2) |
| Expected-organ table | `EXPECTED` in `scripts/validate_anatomy.py` as of this commit (post T11/T12 anatomy fix), unchanged |
| Ranking | ascending minimum surface distance (the α/β blend reduces to this; overlap ≡ 0 after carve-out) |

### Runs

| Name | Flags | Role |
|---|---|---|
| `lesion_validation_v2` | `--calibration scale --crossfit … --organ_stats …` | **Primary** |
| `lesion_validation_negatives_v2` | primary + `--negatives` | Correction of an already-reported variant |
| `lesion_validation_zscore_v2` | `--calibration zscore` (floor 2.0) + crossfit | Correction of an already-reported variant |

The two variants are reruns of configurations already reported in the paper, so
their published numbers can be corrected. They are not new comparisons and no
new variant is added.

## Endpoints

- **Primary:** mean rank of the first expected organ vs chance, 95% CI by
  patient-clustered bootstrap, one-sided permutation p-value, from
  `scripts/analyze_lesions.py` exactly as in the submitted paper.
- **Also reported, all 180 lesions:** top-1 and top-3.
- **Descriptive only:** per-level table, which classes win top rank and how often
  correctly (bladder, femurs), impossible top-1 count, `n_ranked`.
- **Exploratory, no p-values:** subgroups by sex, age (cohort median split),
  primary cancer (groups with at least 5 patients, others pooled), lesion type
  (Reviewer 3, Q7).

## Commitments

- No configuration change after seeing results. No retuning of floor, k, stride
  or EXPECTED.
- If the primary weakens or loses significance, the paper says so, and the
  downstream section is presented as exploratory (Reviewer 5 already asks for
  that framing given prior development on this cohort).
- The Discussion's "two broken classes" framing is re-derived from the new
  per-class counts, not carried over.

## Phase 1: equal-supervision baseline and threshold provenance (Reviewer 5, pt 1)

`scripts/linear_probe.py`: a linear probe (one `nn.Linear(768, 11)`) on frozen
**pretrained** BioMedCLIP patch tokens (mean of blocks 8�11, as the inference
engine fuses them), trained on the exact 14�14 targets OrganPrompt's patch loss
uses (area-pool, > 0.5). Same seed-42 volume split: train 931 volumes / 35,206
items, val 115 / 4,561, test 120 / 4,506. Fixed in advance: AdamW lr 1e-3,
wd 1e-4, batch 64, 5 epochs, seed 0, best epoch by validation loss. No
hyperparameter search.

Both the probe and OrganPrompt (v5 epoch 11, unchanged) are scored on the full
test split with per-class IoU-optimal thresholds **tuned on the validation
split**. The submitted 0.653 used thresholds swept on the test split itself;
each method is also reported under test-tuned ("oracle") thresholds, and
OrganPrompt's oracle row must reproduce 0.653. Primary comparison: val-tuned
mIoU, probe vs OrganPrompt, plus per-class IoU.

## Results

(Appended after the runs, dated, without editing anything above.)

### Addendum 2026-09-29 (evening): primary run and Phase 1 baseline

**Primary (`results/lesion_validation_v2`, config recorded in its summary.json:
scale calibration, floor 0.6, no negatives, cross-fit folds, organ_stats_all55,
seg_geometry dicom-iop-ipp).** 180 lesions / 46 patients, 178 ranked.

| | Submitted | Revised |
|---|---|---|
| Mean rank [95% CI, patient-clustered] | 1.78 [1.56, 2.03] | 1.82 [1.65, 2.02] |
| Chance | 2.91 | 2.91 |
| Permutation p (20,000) | < 0.0001 | < 0.0001 (0 exceedances) |
| Top-1 / top-3 (all 180) | 0.622 / 0.894 | 0.594 / 0.894 |
| Unexpected organ top-1 (<50 mm) | 34 | 35 |
| Bilateral pair splits | (not recomputed) | 48/540 (femur 38, kidney 10) |

Levels at or below chance: T12 (1.70 vs 1.50), L1 (2.12 vs 2.00), L2 (3.60 vs
3.00), L5 (3.08 vs 3.00) — the same four as the submitted run.

Top-1 by class (wins / expected): bladder 32 / 7; femur-left 11 / 0,
femur-right 10 / 0 (femurs 21 / 0, submitted 14 / 0); pancreas 10 / 1; heart
5 / 5 (submitted 26 / 16); lung-right 41 / 39; lung-left 35 / 31. Bladder is
top-1 at 5/10 L2 and 5/13 L5 lesions.

**Phase 1 equal-supervision baseline (`results/equal_supervision/probe.json`).**
Linear probe, pre-registered config (best epoch = 4 of 5, validation loss
still falling — not converged; reported as run). Test mIoU 0.6457 with
validation-tuned thresholds (0.6472 test-tuned), Dice 0.752, HD95 12.89 px,
NSD 0.524. OrganPrompt with validation-tuned thresholds: pending.

### Addendum 2026-09-29 (22:20): OrganPrompt equal-supervision comparison

`results/equal_supervision/organprompt.json` (v5 epoch 11, unchanged). Oracle
(test-tuned) row reproduces the submitted figure: mIoU 0.6528 (paper 0.653).

| Test split | mIoU | Dice | HD95 px | NSD |
|---|---|---|---|---|
| OrganPrompt, validation-tuned thresholds (primary) | 0.6513 | 0.7564 | 19.56 | 0.5488 |
| Linear probe, validation-tuned thresholds | 0.6457 | 0.7521 | 12.89 | 0.5239 |
| OrganPrompt, test-tuned (submitted protocol) | 0.6528 | 0.7570 | 18.79 | 0.5567 |
| Linear probe, test-tuned | 0.6472 | 0.7531 | 12.81 | 0.5282 |

Per class (validation-tuned), OrganPrompt minus probe IoU: bladder +0.020,
femur-left +0.014, femur-right -0.009, heart +0.019, kidney-left -0.018,
kidney-right -0.045, liver +0.007, lung-left +0.023, lung-right +0.017,
pancreas +0.016, spleen +0.007. No significance test was pre-registered for
this comparison; reported descriptively.

### Addendum 2026-09-30 (morning): variants and checkpoint re-selection

**Variant reruns** (corrected loader + cross-fit, configs as originally pre-registered):

| | Primary (v2) | Negatives v2 | Z-score (k=2) v2 |
|---|---|---|---|
| n ranked / 180 | 178 | 170 | 165 |
| Mean rank [95% CI] | 1.82 [1.65, 2.02] | 2.13 [1.80, 2.53] | 1.90 [1.65, 2.20] |
| Chance (over ranked) | 2.91 | 2.86 | 2.85 |
| Top-1 / top-3 (all 180) | 0.594 / 0.894 | 0.469 / 0.782 | 0.478 / 0.817 |
| Unexpected top-1 (<50 mm) | 35 | 58 | 90 |
| Bilateral pair splits | 48/540 | 75/537 (lung 45, kidney 26, femur 4) | 0/540 |

Same conclusions as the submitted runs: neither variant is adopted.

**Checkpoint re-selection** (`results/equal_supervision/checkpoint_selection.json`):
validation mIoU peaks at epoch 8 (0.6482; epoch 11 = 0.6468). Epoch 8 on test with
validation-tuned thresholds: mIoU 0.6528 (epoch 11: 0.6513). Difference 0.0015 <
the ~0.01 materiality bar set before the run, so epoch 11 is retained for all
downstream work and the re-selection is reported. Epochs 0-7 reproduce the
values of an earlier interrupted log exactly.
