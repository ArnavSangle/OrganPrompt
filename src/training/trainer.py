from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from open_clip.loss import SigLipLoss

from src.training.patch_loss import PatchContrastiveLoss


class Trainer:
    """Fine-tunes BioMedCLIP with SigLipLoss + optional patch-level contrastive loss.

    Args:
        model: BioMedCLIP model with freeze_model() already applied.
        train_loader: DataLoader yielding (image, caption, label, gt_mask) batches.
        val_loader: DataLoader for validation.
        device: "cuda" or "cpu".
        lr: Peak learning rate for AdamW.
        warmup_steps: Number of linear warmup steps.
        checkpoint_dir: Directory to save .pt checkpoints.
        lambda_patch: Weight for patch-level contrastive loss (0 = disabled).
    """

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        lr: float = 1e-4,
        warmup_steps: int = 100,
        checkpoint_dir: str = "checkpoints",
        lambda_patch: float = 0.0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.loss_fn = SigLipLoss()
        self.patch_loss_fn = PatchContrastiveLoss(model, lambda_patch=lambda_patch) if lambda_patch > 0 else None
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=0.01,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )
        self.warmup_steps = warmup_steps
        self._peak_lr = lr
        self._step = 0
        self.best_val_loss = float("inf")

    def _encode_batch(self, images, tokens):
        images = images.to(self.device)
        tokens = tokens.to(self.device)
        image_features = F.normalize(self.model.encode_image(images), dim=-1)
        text_features  = F.normalize(self.model.encode_text(tokens),  dim=-1)
        return image_features, text_features  # patch hook fires during encode_image

    def _warmup_lr(self):
        if self._step < self.warmup_steps:
            scale = self._step / max(1, self.warmup_steps)
            for pg in self.optimizer.param_groups:
                pg["lr"] = scale * self._peak_lr

    def train_epoch(self) -> float:
        self.model.train()
        total_loss, n_batches = 0.0, 0
        n_total = len(self.train_loader)
        for images, captions, _, gt_masks in self.train_loader:
            self._step += 1
            self._warmup_lr()
            self.optimizer.zero_grad()
            image_features, text_features = self._encode_batch(images, captions)
            logit_bias = getattr(self.model, "logit_bias", None)
            logit_scale = self.model.logit_scale.exp()
            loss = self.loss_fn(image_features, text_features, logit_scale, logit_bias)
            if self.patch_loss_fn is not None:
                loss = loss + self.patch_loss_fn(
                    text_features, gt_masks.to(self.device), logit_scale.item()
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0
            )
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            n_batches += 1
            print(f"  batch {n_batches}/{n_total} | loss={total_loss/n_batches:.4f}", flush=True)
        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def val_epoch(self) -> float:
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        for images, captions, *_ in self.val_loader:
            image_features, text_features = self._encode_batch(images, captions)
            logit_bias = getattr(self.model, "logit_bias", None)
            loss = self.loss_fn(
                image_features, text_features,
                self.model.logit_scale.exp(),
                logit_bias,
            )
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(1, n_batches)

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        from src.training.freeze import UNFREEZE_PATTERNS
        path = self.checkpoint_dir / f"epoch_{epoch:03d}_val{val_loss:.4f}.pt"
        torch.save({
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": {
                k: v for k, v in self.model.state_dict().items()
                if any(pat in k for pat in UNFREEZE_PATTERNS)
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def train(self, epochs: int) -> None:
        import time
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs * len(self.train_loader)
        )
        for epoch in range(epochs):
            t0 = time.time()
            train_loss = self.train_epoch()
            val_loss = self.val_epoch()
            elapsed = time.time() - t0
            print(f"Epoch {epoch+1}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f} | {elapsed:.0f}s")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                print(f"  New best checkpoint saved (val={val_loss:.4f})")
