# Bone-Metastasis–Organ Proximity Index — Design Spec

**Date:** 2026-07-23
**Status:** Draft for review
**Builds on:** OrganCAM (text-supervised weakly-supervised organ localization in CT)

## 1. Goal

Extend the OrganCAM patch-similarity pipeline from *organ localization* to a
clinical application: quantify the spatial relationship between **bone
metastases** and **abdominal/thoracic organs** in CT, producing a per-patient
**metastasis index** — a blended overlap + proximity score describing how far
disease has spread toward critical structures.

The organ half reuses the validated OrganCAM v5 model (no pixel labels). The
tumor half is developed in two phases: first from ground-truth lesion masks
(to build and validate the index machinery), then via the same
weakly-supervised patch-contrastive recipe (the on-thesis method extension).

## 2. Background / context

- OrganCAM localizes 11 TotalSegmentator organs from text prompts via
  per-patch BioMedCLIP cosine similarity (`src/explainability/patch_similarity.py`),
  fine-tuned with a patch-contrastive loss (`src/training/patch_loss.py`).
  Current best: mIoU 0.653 (v5).
- The pipeline is inherently **2D**: BioMedCLIP ViT-B/16 operates on 224×224
  axial slices; per-slice maps come from a 14×14 patch grid upsampled to 224.
- The metastasis index requires **3D** reasoning (a lesion in one slice, an
  organ in another). The bridge: run OrganCAM slice-by-slice and **stack the 2D
  organ maps into a pseudo-3D organ probability volume**, then compute the index
  against the 3D tumor mask using real voxel spacing.

## 3. Datasets (verified on disk)

| Dataset | Format | Role | Notes |
|---|---|---|---|
| **Spine-Mets-CT-SEG** | 3D DICOM, 0.5 mm, 360 mm FOV, 55 patients | **Primary anchor** | SEG = per-vertebra masks (T1–L5), one segment per level. Lesion type (Lytic/Blastic/Mixed) is in a SEPARATE clinical file `Spine-Mets-CT-SEG_Clinical.xlsx` (downloaded from TCIA, CC BY 4.0) — columns list affected vertebral levels per patient. Tumor mask = union of SEG masks for the listed levels. FOV verified full-body-width; liver/heart/lung/kidney in-frame. |
| **BM-Seg** | 2D 8-bit JPEG, 1517 slices, 23 patients | Phase-2 extra training data | Lesion + bone masks. HU lost (already windowed); no 3D/spacing → not an index anchor. |
| **CT-ORG** | 3D NIfTI, 140 CT | Organ-side validation | Organ + bone GT (liver/kidney/lung/bladder/bones). |
| **TotalSegmentator** | 3D NIfTI | Organ model training | Already used for OrganCAM v5. |
| PKG-Osteosarcoma | H&E histopathology | **Discard** | Not CT — irrelevant. |

**FOV finding (pivotal):** Spine-Mets `SKIN_TO_SKIN_SIM_0.5MM` series are
360×360 mm reconstructions covering the full body cross-section, not
spine-cropped. Organs are present in the same slices as lesioned vertebrae.

## 4. Method

### 4.1 Tumor localization — staged

- **Phase 1 (GT, works today):** tumor input = Spine-Mets GT lesion mask
  **∩ skeleton mask**. Skeleton derived by HU threshold and/or the vertebral
  segmentation already provided. This is exact; it de-risks the index design.
- **Phase 2 (weakly-supervised, the method):** fine-tune the patch-contrastive
  recipe on Spine-Mets (+ BM-Seg) lesion masks so a lesion **text prompt**
  yields a tumor pseudo-mask — the OrganCAM approach applied to lesions.
  Report the index from *both*: Phase 2 = the method, Phase 1 = the ceiling.

### 4.2 Organ localization

Reuse OrganCAM v5 unchanged. For each axial slice: HU-window, resample to 224,
run `BioMedCLIPPatchSimilarity` for each organ prompt → per-slice organ
probability maps → stack into a 3D organ volume aligned to the CT grid.

### 4.3 The index

For each (lesion T, organ O) at native voxel spacing:

```
overlap(T,O)   = |T ∩ O| / |T|
proximity(T,O) = 1 / (1 + min_surface_distance(T,O)_mm)
score(T,O)     = α·overlap + β·proximity
```

Aggregate → per-organ and per-patient metastasis index (aggregation scheme —
burden-weighted, nearest-organ, or sum — to be fixed during Phase 1 on GT).
α/β justified empirically on the GT phase. Lesion-type labels (lytic/blastic)
may weight the score as an extension.

## 5. Architecture (mirror OrganCAM layout)

```
src/data/spine_mets.py        DICOM loader: CT volume + lesion mask + spacing; slice iterator
src/data/bm_seg.py            (Phase 2) 2D JPEG loader for lesion fine-tuning
src/tumor/skeleton.py         HU-threshold / seg-derived skeleton mask
src/tumor/index.py            overlap + proximity + aggregation (the index core)
src/tumor/organ_volume.py     run OrganCAM per-slice → stacked 3D organ prob volume
scripts/build_organ_volume.py CT → 3D organ probability volume (uses OrganCAM v5)
scripts/compute_index.py      organ volume + tumor mask → metastasis index (Phase 1)
scripts/eval_index.py         validate: tumor vs Spine-Mets GT, organ vs CT-ORG
scripts/finetune_lesion.py    (Phase 2) patch-contrastive fine-tune on lesion masks
```

Each unit is independently testable: the index math (`src/tumor/index.py`) is
pure geometry on numpy arrays and needs no model; the organ volume builder is
a thin wrapper over existing OrganCAM code.

## 6. Data flow

```
Spine-Mets DICOM → spine_mets.py → (CT volume HU, spacing, lesion mask 3D)
CT volume → build_organ_volume.py (OrganCAM v5, per-slice) → organ prob volume 3D
lesion mask ∩ skeleton → tumor mask 3D  [Phase 1]
(organ volume, tumor mask, spacing) → index.py → per-patient metastasis index
```

## 7. Validation

- **Tumor half:** Phase-2 pseudo-masks vs Spine-Mets GT lesion masks
  (IoU/Dice per lesion, reusing OrganCAM's metric suite).
- **Organ half:** OrganCAM organ maps on Spine-Mets/CT-ORG slices vs CT-ORG
  organ GT (sanity that organ localization holds on a new scanner/protocol).
- **Index:** Phase 1 (GT) is the reference; Phase 2 index correlates against it.
  Report per-organ and per-patient scores; qualitative overlays.

## 8. Scope

**In scope (Phase 1, first cut):** Spine-Mets loader, skeleton isolation, organ
volume builder, the blended index, GT-based validation, qualitative overlays.

**Phase 2 (follow-on):** lesion patch-contrastive fine-tune, weakly-supervised
tumor pseudo-masks, index-vs-ceiling comparison, BM-Seg as extra training data.

**Out of scope:** new organ classes, higher-res patch grid (accepted 14×14
constraint), 3D CNN training, PET.

## 9. Risks / open questions

- **Organ-model transfer:** OrganCAM was trained on TotalSegmentator CT; Spine-Mets
  is RT-sim CT (different protocol/contrast). Organ maps may degrade → the CT-ORG
  organ validation checks this before trusting the index.
- **Aggregation scheme + α/β:** not yet fixed; to be chosen empirically on GT.
- **Slice alignment:** OrganCAM resamples slices to 224; organ volume must be
  resampled back to native grid for correct 3D distances.
- **Phase 2 success not guaranteed:** bone mets are small/low-contrast (cf.
  pancreas ceiling). Phase 1 does not depend on Phase 2 — deliberate de-risking.
- **Clinical framing:** the index is a spatial/proximity descriptor, not a
  validated clinical metastasis-risk score; positioning must stay honest.
