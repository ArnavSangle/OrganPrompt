import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


class BioMedCLIPPatchSimilarity:
    """Patch-text similarity saliency map for BioMedCLIP.

    Hooks model.visual.trunk.norm to capture all 197 token features after the
    final LayerNorm. Projects each of the 196 patch tokens through the visual
    head (768→512), then computes:

        CAM = sim(patch, positive_prompt) - max(sim(patch, negative_prompts))

    Returns a (image_size, image_size) float32 heatmap in [0, 1].
    No backward pass required — pure forward.
    """

    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self._token_features: torch.Tensor | None = None
        self._hook = None
        self._register_hook()

    def _register_hook(self) -> None:
        def _forward(module, input, output):
            self._token_features = output.detach()  # (1, 197, 768)

        self._hook = self.model.visual.trunk.norm.register_forward_hook(_forward)

    def remove_hooks(self) -> None:
        if self._hook:
            self._hook.remove()
            self._hook = None

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        text_features: torch.Tensor,
        target_class: int,
        neg_text_features: torch.Tensor | None = None,
        image_size: int = 224,
    ) -> np.ndarray:
        """
        Args:
            image: (1, 3, H, W) on self.device
            text_features: (num_classes, 512) L2-normalised, on self.device
            target_class: index into text_features to explain
            neg_text_features: (num_neg, 512) L2-normalised hard negatives
            image_size: spatial size of the returned mask

        Returns:
            mask: (image_size, image_size) float32 array in [0, 1]
        """
        self.model.encode_image(image)  # triggers hook → fills _token_features

        # patch tokens: drop CLS (index 0), project to 512-dim, normalise
        patch_tokens = self._token_features[0, 1:, :]          # (196, 768)
        patch_emb    = self.model.visual.head(patch_tokens)     # (196, 512)
        patch_emb    = F.normalize(patch_emb, dim=-1)           # (196, 512)

        # positive similarity
        cam = patch_emb @ text_features[target_class]           # (196,)

        # subtract max hard-negative similarity
        if neg_text_features is not None:
            neg_sim = (patch_emb @ neg_text_features.T).max(dim=-1).values  # (196,)
            cam = cam - neg_sim

        # reshape to 14×14
        grid = int(cam.shape[0] ** 0.5)   # 14 for ViT-B/16 at 224px
        cam  = cam.reshape(grid, grid)

        # normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        # bilinear upsample to image_size
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        return cam.astype(np.float32)


def post_process_mask(
    mask: np.ndarray,
    threshold: float = 0.4,
    min_size: int = 200,
    fill_holes: bool = True,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """Convert a soft [0, 1] saliency map into a clean binary mask.

    Steps:
      1. Threshold → binary
      2. Remove connected components smaller than min_size pixels
      3. Fill enclosed holes
      4. Optional Gaussian smooth (re-normalised to [0, 1])

    Returns float32 [0, 1].
    """
    binary = mask >= threshold

    if min_size > 0:
        labeled, n = ndimage.label(binary)
        sizes = ndimage.sum(binary, labeled, range(1, n + 1))
        for i, sz in enumerate(sizes):
            if sz < min_size:
                binary[labeled == (i + 1)] = False

    if fill_holes:
        binary = ndimage.binary_fill_holes(binary)

    if smooth_sigma > 0:
        smoothed = ndimage.gaussian_filter(binary.astype(np.float32), sigma=smooth_sigma)
        peak = smoothed.max()
        return (smoothed / peak).astype(np.float32) if peak > 0 else smoothed

    return binary.astype(np.float32)
