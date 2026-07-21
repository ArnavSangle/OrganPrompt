import torch
import open_clip

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

class BioMedCLIP:
    """Thin wrapper around BioMedCLIP loaded via open_clip.

    All tensors are returned on CPU and L2-normalized (unit vectors)
    so dot product == cosine similarity.
    """

    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, 224, 224) float tensor -> (B, D) L2-normalized features."""
        features = self.model.encode_image(images)
        return torch.nn.functional.normalize(features, dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """texts: list of B strings -> (B, D) L2-normalized features."""
        tokens = self.tokenizer(texts)
        features = self.model.encode_text(tokens)
        return torch.nn.functional.normalize(features, dim=-1)

    @torch.no_grad()
    def similarity(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        """Return (n_images, n_texts) cosine similarity matrix."""
        img_features = self.encode_image(images)
        txt_features = self.encode_text(texts)
        return img_features @ txt_features.T
