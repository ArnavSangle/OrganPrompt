# OrganPrompt

Weakly-supervised organ localization in CT from text prompts, via fine-tuned BioMedCLIP patch-similarity. Two phases: a frozen training pipeline (Phase 1) and an active downstream application (Phase 2, current work).

## Project map

```
Phase 0 — OrganAMNIST prototype           DELETED (proved the idea, then removed)
       |
       v
Phase 1 — TotalSegmentator training       FROZEN, checkpoint v5, mIoU 0.653
  Grad-CAM -> patch-similarity engine (src/explainability/patch_similarity.py)
  patch-contrastive loss + multi-layer fusion (ViT blocks 8-11)
  caption ablation -> FEW_SHOT_TEMPLATES (src/labels/templates.py)
  inference-time calibration (mIoU 0.574 -> 0.633, no retrain)
  checkpoint: checkpoints/totalseg_patch_v5/epoch_011_val3.4584.pt
       |
       v
Phase 2 — Bone-metastasis index            ACTIVE
  data loaders (Spine-Mets DICOM/SEG, src/data/spine_mets.py)
  index math (overlap + spacing-aware proximity, src/tumor/index.py)
  CT-ORG transfer sanity check
  per-patient validation (n=20) -> bilateral-pair fix, calibration-leak fix
  per-lesion validation (n=180) -> SIGNIFICANT, p<0.0001   <- headline result
       |
       v
  Failure diagnosis
  - field-of-view theory        REJECTED (scripts/scan_extent.py)
  - T11/T12 lung ground truth   FIXED (docs/references/anatomy-citations.md)
  - missing-class / global negatives   TESTED, mixed/negative result
  - z-score calibration (k=2)   TESTED, mixed/negative result
       |
       v
  Deciding next steps                       YOU ARE HERE
```

Docs: `docs/plans/2026-07-23-bone-metastasis-index-phase1.md` (implementation plan), `docs/specs/2026-07-23-bone-metastasis-index-design.md` (design spec).

## Current state (Phase 2)

- **Headline result:** per-lesion validation, 180 lesions / 46 patients. After the T11/T12 anatomy fix (below), mean rank **1.78**, 95% CI [1.56, 2.03] (patient-clustered bootstrap), top-1 0.622 / top-3 0.894, chance **2.91**, p < 0.0001. (Chance is a function of the `EXPECTED` set sizes, so the anatomy fix moved it too: 3.01 pre-fix → 2.91 post-fix. An earlier "2.96" in the docs matched neither run and has been corrected.)
- **Per-level structure (post-anatomy-fix):** T1-**T11** all beat chance (T8, T9 perfect at 1.00). **T12, L1, L2, L5** sit at or below chance. L3, L4 beat chance clearly. The failure band is the thoracolumbar junction, not the method broadly. **T11 moved above chance as a consequence of the anatomy fix** (adding lungs to its expected set both fixed the key and enlarged k, lowering chance) — earlier write-ups saying "T11-L2" were stale. `scripts/make_lesion_figures.py` now derives the band from the data instead of hardcoding level names, so it cannot drift again.
- **Failure band is narrower than "T12-L2 + L5" implies** (found 2026-07-29 while drafting the paper Discussion). The four failing levels fail in three different ways: **T12 (+0.10) and L5 (+0.08) are at chance, not below it** — T12 faces the study's most stringent bar (k=7, chance 1.50, so it must rank correct-first nearly every time); **L1 (+0.41)** is a modest miss against a stringent bar (k=5); **L2 (+0.90 against a lenient chance of 3.00, k=3) is the only genuine failure.** Report L2 as the outlier; don't average the four into one band.
- **Two classes cause much of the damage** (same session). Across all 180 lesions: **bladder** top-ranks 29 lesions — **21 wrong / 8 correct**, i.e. it wins ~3x more often where it doesn't belong. **femur-left + femur-right** top-rank 14 lesions — **0 correct**; the femurs never once rank first where a femur is actually expected. This explains both failing levels: bladder hijacks 4/10 lesions at L2 (upper lumbar, bladder nowhere near), and at L5 — the one level where femurs ARE expected — no femur ever wins. Matches `scripts/prompt_confusion.py` exactly (femur prompt ≈ humerus/rib cortical ring; bladder ≈ aorta), and fig3 confirms visually that winning femur patches sit on ribs. **Reframe: this is not "hard levels", it is two broken classes.** Small n per level (10 at L2, 13 at L5) — a diagnosis to act on, not a precise effect size.
- **34/180 impossible top-1 detections**, still concentrated in bladder and femur phantoms.
  - **Field-of-view: rejected.** `scripts/scan_extent.py` shows phantom patients have *more* caudal scan coverage than clean ones (median 122mm vs 88.5mm below the last segmented vertebra), 9/12 reach L5.
  - **T11/T12 lung ground truth: fixed.** The `EXPECTED` table dropped lungs abruptly after T10; anatomical literature (PMID 19589417 + others, cited in `docs/references/anatomy-citations.md`) puts the true costophrenic recess extent at T11 or below, never above. Fixed in `scripts/validate_anatomy.py`. This alone moved impossible top-1 from 38→34 and mean rank from 1.94→1.78 — the fix was decided from the literature first, then applied, not tuned to the number.
  - **Missing-class / global hard negatives: tested, mixed-negative result.** Adding rib/humerus/aorta as `NEGATIVE_PROMPTS` (motivated by `scripts/prompt_confusion.py`, a text-encoder-only diagnostic showing femur's prompt embeds almost identically to humerus/scapula, bladder's to aorta) moved mean rank 1.94→2.04 and top1/top3 both down, with impossible top-1 roughly flat (37 vs 38). Femur bilateral pair-splitting improved sharply (31→4) but a **new** lung pair-splitting problem appeared (0→46/540) — the rib negative, applied globally, likely bled into lung's own scoring at the rib-lung boundary. Not adopted as-is. Results: `results/lesion_validation_negatives/summary.json`.
  - **Z-score calibration (k=2, fixed in advance): tested, mixed-negative result.** Swapping the ratio-to-peak floor for a per-organ z-score floor (`mean + 2*sd`) moved mean rank 1.94→1.82 (better) and bilateral pair splits to **zero** (perfect), but impossible top-1 **more than doubled** (38→79) — nearly every wrong detection wins at 0-1mm, consistent with the z-score floor being too permissive (most organs retain a sliver of territory almost everywhere instead of correctly vanishing). Not adopted as-is. Results: `results/lesion_validation_zscore/summary.json` (baseline reproduction: `results/lesion_validation_scale_check/summary.json`).
  - **Caveat on both variants' mean ranks:** they leave more lesions *unrankable* than the baseline (`n_ranked` 178 baseline, 171 negatives, 164 z-score), so their mean ranks average over a smaller, easier subset, which flatters that metric. top1/top3 are computed over all 180 and are the cleaner comparison — both variants fall there. Quote top1/top3 when comparing variants, not mean rank alone.

## Future steps, roughly in order

1. **Decide on a more surgical negative-prompt retry.** The global-negative failure has a specific diagnosed cause (rib bleeding into lung because it's subtracted from every organ, not just femur). A per-organ-scoped version — humerus/rib only against femur-left/right, aorta only against bladder — targets that specific mechanism and hasn't been tried. On hold pending user direction; not to be launched without discussion given eval-set fatigue.
2. **z-score calibration is not worth adopting as-is** given the doubled impossible-top-1 rate; a smaller/tuned k was explicitly not tried (k=2 was pre-registered). Revisiting this would need fresh pre-registration, not a retune of the same run.
3. **Decide if Phase 2 is paper-ready as-is.** p<0.0001 on 180 lesions/46 patients, an honest calibration history, a resolved ground-truth bug, and two documented (if imperfect) calibration experiments may be sufficient. Chasing every remaining phantom risks the multiple-comparisons problem already flagged once. This call is the user's, not something to auto-decide.
4. **Bone-tumor + metastasis-index dataset extension** (separate, lower priority) — same algorithm applied to a new tissue target; paper strategy TBD.

## Standing rules

- **Never push to GitHub.** User controls all pushes; commit only when asked.
- **No `Co-Authored-By` trailer or any AI/Claude mention** in commit messages or history.
- **Do not suggest higher-resolution input.** Upsampling breaks mask generation; the 14x14 ViT-B/16 patch grid is an accepted architectural constraint.
- **No GPU/CPU restriction.** Run any script — including training — on whichever device is most time-efficient. Only real constraint: watch for VRAM contention when multiple runs are concurrent, and serialize if that's a genuine risk (not a policy issue, just a resource one).
- **Pre-register experiments on the evaluation set.** Decide the configuration before seeing results; report whatever comes back. The eval set has already absorbed several comparisons (calibration fit, bilateral grouping, per-lesion vs per-patient) — treat further tests on it as a scarce resource.
- **Do not tune ground truth to results.** If an `EXPECTED` table or similar looks wrong, settle it from an independent (anatomical) reference before touching it, never by checking which change improves the metric.
- **Rejected explanations, don't re-litigate without new evidence:** a min-size filter cannot remove phantoms (correct organs survive at sizes smaller than phantoms); field-of-view does not explain the bladder/femur phantoms (checked directly, phantom patients have more coverage, not less); global hard negatives and z-score calibration (k=2) both tested and not adopted (see Current state above) — don't re-run either without a new, more specific hypothesis.
- **Verify subagent process claims directly** (`tasklist`/`Get-Process`/`nvidia-smi`), don't trust self-reports. A subagent instructed to switch `--device` mid-task may refuse, silently launch a second competing process, or revert to its original instruction even after explicit supersession — check the actual OS process and command line before trusting any "done" or "running" claim.
- **Do not use the Agent tool, workflows, or deep-research unless explicitly requested.**

## Key files

- `src/labels/templates.py` — `CLASS_NAMES` (11 organs, dict keyed by int), `FEW_SHOT_TEMPLATES` (the prompts actually used), `NEGATIVE_PROMPTS`.
- `src/tumor/assign.py` — competitive cross-organ assignment, bilateral pair grouping, `TYPICAL_PEAK` calibration constants (refit on 35 held-out patients — see comment in file for provenance).
- `src/tumor/index.py` — overlap + proximity scoring per lesion.
- `src/data/spine_mets.py` — Spine-Mets DICOM/SEG loader, per-vertebra lesion coordinates.
- `src/data/scan_extent.py` — header-only scan coverage measurement (no pixels, no model).
- `scripts/validate_lesions.py` / `scripts/analyze_lesions.py` — the per-lesion validation pipeline and its patient-clustered statistics.
- `scripts/scan_extent.py` — field-of-view diagnostic (rejected that hypothesis).
- `scripts/prompt_confusion.py` — text-encoder-only diagnostic for the missing-class hypothesis.
- `scripts/fit_organ_stats.py` — refits `TYPICAL_PEAK` on held-out patients (regenerate this, not the old `fit_organ_peaks.py`, if calibration needs updating).
- `docs/references/anatomy-citations.md` — anatomical sources backing the T11/T12 lung ground-truth fix, for the paper's methods section.
- `src/labels/templates.py` — also now has `NEGATIVE_PROMPTS` entries for rib, humerus, aorta (tested, not adopted as default — see Current state); `validate_lesions.py` / `organ_volume.py` support them via `--calibration`-independent `neg_text_features` threading, opt-in via `load_organcam(..., with_negatives=True)`.
- `src/tumor/assign.py` — `CompetitiveAssigner(..., calibration="scale"|"zscore")` — `scale` is the unchanged default; `zscore` is the tested-not-adopted alternative floor.

## Memory

Additional context — user preferences, past findings, and project history that predates or supplements this file — lives in the auto-memory system (`MEMORY.md` and linked files), not duplicated here.
