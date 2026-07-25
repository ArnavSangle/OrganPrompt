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


def _organ_similarity(ct, explainer, text_features, organ_class, device, label=""):
    """Similarity volume [D,H,W] for ONE organ across every axial slice."""
    D, H, W = ct.shape
    out = np.zeros((D, H, W), dtype=np.float32)
    for zi in range(D):
        img = preprocess_slice(ct[zi], organ_class).unsqueeze(0).to(device)
        m = explainer.generate(img, text_features, target_class=organ_class, image_size=224)
        t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
        out[zi] = t.squeeze().numpy()
        if (zi + 1) % 50 == 0 or zi + 1 == D:
            print(f"[organ_volume]   {label} slice {zi + 1}/{D}", flush=True)
    return out


def iter_organ_similarity(ct, model, text_features, device, organs=None):
    """Yield (organ_class, [D,H,W] similarity) one organ at a time.

    Streaming matters on full uncropped volumes: the stacked [organs,D,H,W] float32
    array is several GB on its own, and callers only ever need one organ resident at
    a time to reduce it to a mask. Use this instead of build_organ_volume whenever
    the volume is not tightly cropped.
    """
    if organs is None:
        organs = list(range(len(CLASS_NAMES)))
    D = ct.shape[0]
    print(f"[organ_volume] streaming {len(organs)} organs x {D} slices ...", flush=True)
    explainer = BioMedCLIPPatchSimilarity(model, device=device)
    try:
        for oi, oc in enumerate(organs):
            label = f"({oi + 1}/{len(organs)}) {CLASS_NAMES[oc]}"
            print(f"[organ_volume] {label} ...", flush=True)
            yield oc, _organ_similarity(ct, explainer, text_features, oc, device, label)
    finally:
        explainer.remove_hooks()


def build_organ_volume(ct, model, text_features, device, organs=None):
    """Run OrganCAM per axial slice, stack into a [organs,D,H,W] similarity volume.

    Materializes every organ at once -- only safe on cropped volumes. Prefer
    iter_organ_similarity for full-resolution scans.
    """
    if organs is None:
        organs = list(range(len(CLASS_NAMES)))
    out = np.zeros((len(organs), *ct.shape), dtype=np.float32)
    for oi, (_, sim) in enumerate(
            iter_organ_similarity(ct, model, text_features, device, organs)):
        out[oi] = sim
    return out
