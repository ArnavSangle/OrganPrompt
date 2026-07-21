import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


class BioMedCLIPPatchSimilarity:
    """Patch-text similarity saliency map using multi-layer feature fusion.

    Hooks model.visual.trunk.blocks[8..11] to capture patch token features
    from 4 ViT blocks. Averages the 196 patch tokens across all 4 layers,
    projects through the visual head (768→512), then computes:

        CAM = sim(patch, positive_prompt) - max(sim(patch, negative_prompts))

    Multi-layer fusion combines semantic responses (blocks 10-11) with
    fine-grained texture/edge responses (blocks 8-9) for better boundary
    localization than using the final norm layer alone.

    Returns a (image_size, image_size) float32 heatmap of RAW similarity
    values (roughly [-0.3, +0.3] for a fine-tuned model). No per-image
    normalization — per-class thresholds derived from a calibration set
    (see scripts/analyze_similarities.py) are applied downstream.
    """

    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self._layer_features: list[torch.Tensor] = []
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        for block_idx in [8, 9, 10, 11]:
            block = self.model.visual.trunk.blocks[block_idx]

            def _forward(module, input, output, _idx=block_idx):
                self._layer_features.append(output.detach())  # (1, 197, 768)

            self._hooks.append(block.register_forward_hook(_forward))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

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
        self._layer_features.clear()
        self.model.encode_image(image)  # triggers 4 hooks → fills _layer_features

        # Stack: (4, 1, 197, 768) → mean → (197, 768) → drop CLS → (196, 768)
        stacked = torch.stack(self._layer_features, dim=0)   # (4, 1, 197, 768)
        fused   = stacked.mean(dim=0)[0]                     # (197, 768)
        patch_tokens = fused[1:, :]                          # (196, 768) — drop CLS

        patch_emb = self.model.visual.head(patch_tokens)     # (196, 512)
        patch_emb = F.normalize(patch_emb, dim=-1)           # (196, 512)

        # positive similarity
        cam = patch_emb @ text_features[target_class]        # (196,)

        # subtract max hard-negative similarity
        if neg_text_features is not None:
            neg_sim = (patch_emb @ neg_text_features.T).max(dim=-1).values  # (196,)
            cam = cam - neg_sim

        # reshape to 14×14
        grid = int(cam.shape[0] ** 0.5)    # 14 for ViT-B/16 at 224px
        cam  = cam.reshape(grid, grid)

        # bilinear upsample to image_size — return raw similarities, no normalization
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        return cam.astype(np.float32)


class BioMedCLIPGradCAM:
    """Gradient-weighted Class Activation Map for BioMedCLIP (ViT-B/16).

    Backprops image-text cosine similarity through the ViT to get
    gradient-weighted patch activation maps at block 11.

        CAM = ReLU( mean_channel(grads) · acts ) for the 196 patch tokens

    Unlike patch-text similarity, this uses the global CLS-token alignment
    the model learned during training, so it works even when patch tokens
    lack explicit spatial discrimination.

    Requires one backward pass per image (slower than BioMedCLIPPatchSimilarity).
    """

    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        block = self.model.visual.trunk.blocks[11]

        def _fwd(module, input, output):
            self._activations = output  # (1, 197, 768)

        def _bwd(module, grad_input, grad_output):
            self._gradients = grad_output[0]  # (1, 197, 768)

        self._hooks.append(block.register_forward_hook(_fwd))
        self._hooks.append(block.register_full_backward_hook(_bwd))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(
        self,
        image: torch.Tensor,
        text_features: torch.Tensor,
        target_class: int,
        image_size: int = 224,
    ) -> np.ndarray:
        """
        Args:
            image: (1, 3, H, W) on self.device
            text_features: (num_classes, 512) L2-normalised, on self.device
            target_class: index into text_features to explain
            image_size: spatial size of the returned mask

        Returns:
            mask: (image_size, image_size) float32 array in [0, 1]
        """
        self.model.zero_grad()
        image = image.detach().requires_grad_(True)

        image_features = F.normalize(self.model.encode_image(image), dim=-1)
        score = (image_features @ text_features[target_class]).sum()
        score.backward()

        # Vanilla gradient: d(score)/d(image) — no hooks, works if any gradient flows
        # Sum absolute gradient across colour channels → (1, H, W) → (H, W)
        cam = image.grad.abs().sum(dim=1).squeeze()  # (224, 224)

        # normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy().astype(np.float32)


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
