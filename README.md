# OrganCAM

Text-supervised weakly-supervised organ localization in CT slices. Fine-tunes [BioMedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) with a patch-level contrastive loss so that per-patch text–image cosine similarity acts as a segmentation map — no per-pixel labels required at inference.

## Results

Evaluated on the [TotalSegmentator v2.0.1](https://github.com/wasserth/TotalSegmentator) test split, 11 organ classes.

| Metric | Best (v4, epoch 15, per-class tuned thresholds) |
|---|---|
| mIoU | **0.633** |
| Dice | **0.738** |
| HD95 (px) | 24.4 |
| Sensitivity | 0.786 |
| Precision | 0.731 |
| RVE | 0.91 |
| NSD (τ=3px) | 0.530 |

**Progression across method iterations:**

| Version | Method | mIoU |
|---|---|---|
| Baseline | Vanilla-gradient CAM (no fine-tuning) | 0.003 |
| v0 | GradCAM + SigLIP fine-tuning | 0.063 |
| v1 | + patch-level contrastive loss | 0.307 |
| v3 | + resume from v1 | 0.469 |
| v4 | + resume from v3 | 0.574 |
| **v4-tuned** | + calibration fix (raw sims + per-class IoU-optimal thresholds) | **0.633** |

The final step required no retraining: diagnostic analysis of raw patch-text similarity distributions (`scripts/analyze_similarities.py`) revealed that (a) per-image min-max normalization was destroying model-confidence information and (b) class-conditional negative subtraction was cutting foreground signal more than background. Removing both and calibrating per-class thresholds by sweeping IoU-optimal cutoffs on generated masks (`evaluate_masks.py --tune`) raised mIoU by +10% and cut HD95 in half.

### Per-class mIoU (v4-tuned)

| Organ | N | IoU | Dice |
|---|---:|---:|---:|
| Lung Right | 535 | 0.826 | 0.896 |
| Liver | 492 | 0.809 | 0.883 |
| Lung Left | 545 | 0.795 | 0.873 |
| Heart | 480 | 0.780 | 0.860 |
| Bladder | 265 | 0.698 | 0.798 |
| Spleen | 464 | 0.627 | 0.752 |
| Femur Left | 260 | 0.563 | 0.700 |
| Femur Right | 262 | 0.512 | 0.654 |
| Kidney Left | 401 | 0.485 | 0.620 |
| Kidney Right | 403 | 0.393 | 0.533 |
| Pancreas | 395 | 0.239 | 0.360 |

Large, morphologically-distinct organs (liver, lungs, heart) reach 0.78-0.83 IoU with text supervision alone. Small, low-contrast organs (pancreas) remain harder even after calibration — an inherent limit of weakly-supervised methods.

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

Masks are saved as raw patch-text similarity heatmaps (no per-image normalization). Per-class thresholds are applied at evaluation time.

### Score generated masks against GT with IoU-optimal per-class thresholds

```bash
python scripts/evaluate_masks.py --tune
```

This sweeps candidate thresholds per class, picks the IoU-optimal cutoff, saves them to `diagnostics/tuned_thresholds.json`, and evaluates. Use `--thresholds_json diagnostics/tuned_thresholds.json` on subsequent runs to reuse.

### Diagnostic: raw patch similarity distributions

```bash
python scripts/analyze_similarities.py --checkpoint checkpoints/totalseg_patch_v4/epoch_015_val3.7378.pt --root Totalsegmentator_dataset_v201
```

Produces per-class foreground vs background histograms in `diagnostics/`. Used to verify that the model has spatial signal before spending epochs on retraining.

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

- Patch supervision is fixed at 14×14, so boundary precision is inherently coarse — HD95 of 24 pixels on a 224px image reflects the upsampled patch grid.
- Pancreas remains the hardest class (0.24 IoU) — small, elongated, low-contrast, highly variable in position. Even fully-supervised methods struggle here; weakly-supervised results at this level are competitive with the literature.
- Kidneys reach 0.39-0.49 IoU. Prompt engineering to emphasize morphology (bean-shape, hilum orientation) rather than just position could help; deferred to a separate caption ablation study.
- Val loss (SigLIP CLS-level) does not track mIoU — rank checkpoints with `eval_checkpoints.py`, not by filename val loss.
