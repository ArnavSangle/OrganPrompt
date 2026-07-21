# OrganCAM

Text-supervised weakly-supervised organ localization in CT slices. Fine-tunes [BioMedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) with a patch-level contrastive loss so that per-patch text–image cosine similarity acts as a segmentation map — no per-pixel labels required at inference.

## Results

Evaluated on the [TotalSegmentator v2.0.1](https://github.com/wasserth/TotalSegmentator) test split, 11 organ classes.

| Metric | Best checkpoint (v4, epoch 15) |
|---|---|
| mIoU | **0.574** |
| Dice | **0.686** |
| HD95 (px) | 51.0 |
| Sensitivity | 0.954 |
| Precision | 0.598 |
| RVE | 32.1 |
| NSD (τ=3px) | 0.373 |

**Progression across architectural iterations:**

| Version | Method | mIoU |
|---|---|---|
| Baseline | Vanilla-gradient CAM (no fine-tuning) | 0.003 |
| v0 | GradCAM + SigLIP fine-tuning | 0.063 |
| v1 | + patch-level contrastive loss | 0.307 |
| v3 | + resume from v1 | 0.469 |
| v4 | + resume from v3 | **0.574** |

### Per-class mIoU (v4)

| Organ | N | IoU | Dice |
|---|---:|---:|---:|
| Liver | 492 | 0.782 | 0.864 |
| Heart | 480 | 0.763 | 0.847 |
| Lung Right | 535 | 0.765 | 0.855 |
| Lung Left | 545 | 0.753 | 0.847 |
| Bladder | 265 | 0.663 | 0.766 |
| Spleen | 464 | 0.480 | 0.624 |
| Femur Left | 260 | 0.449 | 0.596 |
| Femur Right | 262 | 0.386 | 0.533 |
| Kidney Right | 403 | 0.265 | 0.392 |
| Kidney Left | 401 | 0.263 | 0.390 |
| Pancreas | 395 | 0.124 | 0.209 |

### Qualitative masks

![Pseudo-mask summary](figures/totalseg/totalseg_summary.png)

## Architecture

- **Backbone**: BioMedCLIP (ViT-B/16 visual encoder + PubMedBERT text encoder), 195M params
- **Fine-tuning**: ViT blocks 8-11 + BERT layers 8-11 + `visual.head` + `text.proj` + `logit_scale` + `logit_bias`
- **Loss**: `SigLipLoss` (global CLS-level contrastive) + λ · patch-level BCE
- **Patch-level contrastive loss**: forward hook on ViT block 11 captures 196 patch tokens per image, projects through `visual.head` (768→512), computes per-patch cosine similarity against the organ text embedding, then BCE against the GT segmentation mask downsampled to the 14×14 patch grid
- **Inference**: multi-layer patch-similarity fusion over ViT blocks 8-11, with negative-prompt subtraction to suppress background tissue; upsamples 14×14 similarity map to 224×224

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install open_clip_torch nibabel scipy matplotlib pytest
```

Download the [TotalSegmentator dataset v2.0.1](https://zenodo.org/records/10047292) and unzip to `Totalsegmentator_dataset_v201/` in the project root.

## Usage

### Precompute slices (one-time, GPU-bound training prep)

```bash
python scripts/precompute_slices.py --root Totalsegmentator_dataset_v201
```

### Fine-tune

```bash
python scripts/train.py --root Totalsegmentator_dataset_v201 --epochs 20 --lambda_patch 0.5
```

Resume from a prior checkpoint (used to build v3 → v4):
```bash
python scripts/train.py --root Totalsegmentator_dataset_v201 --epochs 20 --resume_from checkpoints/totalseg_patch_v3/epoch_016_val3.5591.pt --checkpoint_dir checkpoints/totalseg_patch_v4
```

### Rank checkpoints across all epochs

```bash
python scripts/eval_checkpoints.py --checkpoint_dir checkpoints/totalseg_patch_v4 --root Totalsegmentator_dataset_v201 --limit 500
```

Writes `checkpoint_metrics.json` with IoU / Dice / HD95 / Sensitivity / Precision / RVE / NSD per epoch.

### Generate pseudo-masks from best checkpoint

```bash
python scripts/generate_masks.py --checkpoint checkpoints/totalseg_patch_v4/epoch_015_val3.7378.pt --root Totalsegmentator_dataset_v201 --method patch_sim
```

### Score generated masks against GT

```bash
python scripts/evaluate_masks.py --masks_dir data/totalseg_masks
```

### Visualize

```bash
python scripts/visualize_masks.py
```

## Project structure

```
src/
  data/           TotalSegmentator NIfTI loader, ImageTextDataset wrapper
  labels/         CLASS_NAMES, FEW_SHOT_TEMPLATES, NEGATIVE_PROMPTS
  training/       Trainer, freeze schedule, PatchContrastiveLoss
  explainability/ BioMedCLIPPatchSimilarity (inference-time mask generator)
scripts/
  precompute_slices.py  cache CT slices to .npy for GPU-bound training
  train.py              fine-tuning entrypoint (patch loss + SigLIP)
  eval_checkpoints.py   rank all epochs by full metric suite
  generate_masks.py     batch pseudo-mask generation
  evaluate_masks.py     score masks against TotalSegmentator GT
  visualize_masks.py    per-organ overlay figures
figures/
  totalseg/       per-class + summary pseudo-mask overlays
checkpoints/      per-epoch .pt files (gitignored, ~200MB each)
```

## Dataset

TotalSegmentator v2.0.1: 1,228 abdominal/thoracic CT volumes with per-organ segmentation labels. Volume-level 80/10/10 split. Slices sampled at top-K per organ per volume (default K=5). HU windowing applied for soft-tissue contrast, resampled to 224×224.

Checkpoints, generated masks, and the raw dataset (~50GB total) are excluded from this repo — run the scripts above to reproduce.

## Known limitations

- Patch supervision is fixed at 14×14, so boundary precision is inherently coarse. Predictions are high-recall (Sens 0.95) but over-predict volume (Prec 0.60, RVE 32) — the model finds where the organ is but not exactly where it ends.
- Kidneys and pancreas underperform (mIoU < 0.30). These are smaller, less-distinctive organs where the coarse patch grid struggles most.
- Val loss (SigLIP CLS-level) does not track mIoU — checkpoints must be ranked with `eval_checkpoints.py` rather than by filename val loss.
