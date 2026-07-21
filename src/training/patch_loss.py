import torch
import torch.nn.functional as F


class PatchContrastiveLoss:
    """Patch-level contrastive loss using GT segmentation masks.

    Registers a forward hook on model.visual.trunk.blocks[11] to capture
    patch token activations as a side effect of the normal forward pass.
    Projects them through model.visual.head (768→512), then computes a
    per-patch BCE loss against the GT mask downsampled to the 14×14 patch grid.

    Positive patches (organ present) are pulled toward the organ text embedding.
    Negative patches (organ absent) are pushed away.

    Usage:
        patch_loss_fn = PatchContrastiveLoss(model, lambda_patch=0.5)
        # inside training loop, after encode_image has been called:
        loss = global_loss + patch_loss_fn(text_features, gt_masks, logit_scale)
        # call remove_hook() when done training
    """

    def __init__(self, model, lambda_patch: float = 0.5):
        self.model = model
        self.lambda_patch = lambda_patch
        self._raw_patch_tokens: torch.Tensor | None = None
        self._hook = None
        self._register_hook()

    def _register_hook(self) -> None:
        block = self.model.visual.trunk.blocks[11]

        def _fwd(module, input, output):
            self._raw_patch_tokens = output[:, 1:, :]  # (B, 196, 768) — drop CLS

        self._hook = block.register_forward_hook(_fwd)

    def remove_hook(self) -> None:
        if self._hook:
            self._hook.remove()
            self._hook = None

    def __call__(
        self,
        text_features: torch.Tensor,  # (B, 512) L2-normalised, already on device
        gt_masks: torch.Tensor,        # (B, 224, 224) float in {0, 1}, on device
        logit_scale: float = 1.0,
    ) -> torch.Tensor:
        B = self._raw_patch_tokens.shape[0]

        # Project patch tokens → (B, 196, 512), L2-normalise
        patch_emb = F.normalize(
            self.model.visual.head(self._raw_patch_tokens), dim=-1
        )

        # Downsample GT masks to 14×14 patch grid → binary (B, 196) labels
        patch_labels = F.interpolate(
            gt_masks.unsqueeze(1),   # (B, 1, 224, 224)
            size=(14, 14),
            mode="area",             # average pooling avoids aliasing
        ).squeeze(1).reshape(B, -1)  # (B, 196)
        patch_labels = (patch_labels > 0.5).float()

        # Per-patch cosine similarity to own-class text embedding: (B, 196)
        sims = torch.einsum("bpd,bd->bp", patch_emb, text_features) * logit_scale

        loss = F.binary_cross_entropy_with_logits(sims, patch_labels)
        return self.lambda_patch * loss
