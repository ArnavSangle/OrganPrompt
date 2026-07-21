from pathlib import Path
import torchvision.transforms as T
from medmnist import OrganAMNIST


class GrayscaleToRGB:
    """Repeat single-channel tensor to 3 channels. Picklable unlike T.Lambda."""
    def __call__(self, x):
        return x.repeat(3, 1, 1)


DATA_ROOT = Path(__file__).parent.parent.parent / "data" / "organamnist"

def load_organamnist(size: int = 224):
    """Return (train, val, test) MedMNIST datasets at the given pixel size.

    size=28: returns (1, 28, 28) single-channel tensors (native grayscale).
    size>=64: returns (3, size, size) tensors with grayscale replicated to 3
    channels so BioMedCLIP's ViT-B/16 image encoder receives the expected input.
    """
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    if size == 28:
        transform = T.Compose([T.ToTensor()])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((size, size), antialias=True),
            GrayscaleToRGB(),
        ])

    train = OrganAMNIST(split="train", transform=transform, download=True, root=str(DATA_ROOT), size=size)
    val   = OrganAMNIST(split="val",   transform=transform, download=True, root=str(DATA_ROOT), size=size)
    test  = OrganAMNIST(split="test",  transform=transform, download=True, root=str(DATA_ROOT), size=size)

    return train, val, test
