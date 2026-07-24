"""Build a 3D organ-probability volume by running OrganCAM per axial slice.

Bridges OrganCAM's inherently-2D patch-similarity inference (224x224, ViT-B/16)
to the 3D geometry the metastasis index needs: each CT slice is preprocessed
exactly as in the OrganCAM training/eval pipeline, run through
BioMedCLIPPatchSimilarity, and the resulting per-slice organ maps are stacked
and downsampled back to the native slice resolution.
"""
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from open_clip import get_tokenizer

from src.data.totalsegmentator import HU_WINDOW
from src.explainability.patch_similarity import BioMedCLIPPatchSimilarity
from src.labels.templates import CLASS_NAMES, FEW_SHOT_TEMPLATES

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def preprocess_slice(hu_slice, organ_class):
    """HU-window (per organ), normalize to [0,1], resize to 224, replicate to 3ch.

    Mirrors TotalSegmentatorDataset's slow-path preprocessing
    (src/data/totalsegmentator.py) so OrganCAM sees the same input distribution
    it was trained on.
    """
    lo, hi = HU_WINDOW.get(organ_class, HU_WINDOW["default"])
    x = np.clip(hu_slice.astype(np.float32), lo, hi)
    x = (x - lo) / (hi - lo)
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)          # (1,1,H,W)
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    return t.squeeze(0).repeat(3, 1, 1)                        # (3,224,224)


def load_organcam(checkpoint, device):
    """Load base BioMedCLIP + fine-tuned checkpoint, encode the 11 organ prompts."""
    print(f"[organ_volume] loading base model {MODEL_NAME} ...", flush=True)
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)
    model = model.to(device)
    print(f"[organ_volume] loading checkpoint {checkpoint} ...", flush=True)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = get_tokenizer(MODEL_NAME)
    prompts = [FEW_SHOT_TEMPLATES[i] for i in range(len(CLASS_NAMES))]
    with torch.no_grad():
        text_features = F.normalize(model.encode_text(tokenizer(prompts).to(device)), dim=-1)
    print("[organ_volume] model ready, text prompts encoded", flush=True)
    return model, text_features


def build_organ_volume(ct, model, text_features, device, organs=None):
    """Run OrganCAM per axial slice, stack into a [organs,D,H,W] similarity volume."""
    if organs is None:
        organs = list(range(len(CLASS_NAMES)))
    D, H, W = ct.shape
    print(f"[organ_volume] running {len(organs)} organs x {D} slices ...", flush=True)
    out = np.zeros((len(organs), D, H, W), dtype=np.float32)
    explainer = BioMedCLIPPatchSimilarity(model, device=device)
    for zi in range(D):
        for oi, oc in enumerate(organs):
            img = preprocess_slice(ct[zi], oc).unsqueeze(0).to(device)
            m = explainer.generate(img, text_features, target_class=oc, image_size=224)
            t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
            out[oi, zi] = t.squeeze().numpy()
        if (zi + 1) % 20 == 0 or zi + 1 == D:
            print(f"[organ_volume] slice {zi + 1}/{D} done", flush=True)
    explainer.remove_hooks()
    return out
