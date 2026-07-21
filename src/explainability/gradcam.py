import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


class BioMedCLIPGradCAM:
    """Grad-CAM on the last ViT block of BioMedCLIP's visual encoder.

    Hooks model.visual.trunk.blocks[-1] to capture the 197-token output
    (1 CLS + 196 patch tokens). On generate(), computes cosine similarity
    between image and target-class text features, backpropagates, and
    returns a (image_size, image_size) heatmap in [0, 1].
    """

    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        target_layer = self.model.visual.trunk.blocks[-1]

        def _forward(module, input, output):
            self._activations = output.detach()

        def _backward(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(target_layer.register_forward_hook(_forward))
        self._hooks.append(target_layer.register_full_backward_hook(_backward))

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
            text_features: (num_classes, D) normalized, on self.device
            target_class: index into text_features to explain
            image_size: spatial size of the returned mask

        Returns:
            mask: (image_size, image_size) float32 array in [0, 1]
        """
        self.model.zero_grad()

        # requires_grad on the image (not params) keeps the graph small
        image = image.detach().requires_grad_(True)
        image_features = F.normalize(self.model.encode_image(image), dim=-1)
        score = (image_features * text_features[target_class]).sum()
        score.backward()

        # activations / gradients: (1, 197, 768)
        # drop CLS token (index 0), keep 196 patch tokens
        acts  = self._activations[0, 1:, :]   # (196, 768)
        grads = self._gradients[0, 1:, :]     # (196, 768)

        # channel-wise gradient mean → weight activations
        weights = grads.mean(dim=0)            # (768,)
        cam = (acts * weights).sum(dim=-1)     # (196,)
        cam = F.relu(cam)

        # reshape to 14×14 spatial grid
        grid = int(cam.shape[0] ** 0.5)        # 14 for ViT-B/16 at 224px
        cam = cam.reshape(grid, grid)

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
