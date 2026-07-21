import random
import torch
from torch.utils.data import Dataset

_ZERO_MASK = torch.zeros(224, 224, dtype=torch.float32)


class ImageTextDataset(Dataset):
    """Wraps a TotalSegmentator dataset and pairs each image with a sampled caption.

    Always returns (image, caption_tokens, label, gt_mask).
    gt_mask is a zero tensor when the base dataset does not provide masks.

    Args:
        base_dataset: dataset returning (image, label) or (image, label, gt_mask)
        captions: dict mapping class_int -> list of pre-tokenised caption tensors
    """

    def __init__(self, base_dataset, captions: dict[int, list]):
        self.base = base_dataset
        self.captions = captions

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        result = self.base[idx]
        if len(result) == 3:
            image, label, gt_mask = result
        else:
            image, label = result
            gt_mask = _ZERO_MASK
        label = int(label[0]) if hasattr(label, '__len__') else int(label)
        caption = random.choice(self.captions[label])
        return image, caption, label, gt_mask
