import os
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image


DATA_ROOT = "data/cifar10"


# ------------------------------------------------------
# Auto-download CIFAR-10 if missing
# ------------------------------------------------------
def ensure_cifar10_exists():
    os.makedirs(DATA_ROOT, exist_ok=True)
    # torchvision will auto-download if data is missing
    return


# ------------------------------------------------------
# Dataset classes
# ------------------------------------------------------
class CIFAR10Train(Dataset):
    def __init__(self, transform=None, root=DATA_ROOT):
        ensure_cifar10_exists()

        # torchvision CIFAR-10 dataset
        self.dataset = CIFAR10(
            root=root,
            train=True,
            download=True,
        )

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class CIFAR10Val(Dataset):
    """
    CIFAR-10 has no official validation split.
    We use the test set as val, matching your TinyImageNet style.
    """
    def __init__(self, transform=None, root=DATA_ROOT):
        ensure_cifar10_exists()

        self.dataset = CIFAR10(
            root=root,
            train=False,
            download=True,
        )

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class CIFAR10Test(Dataset):
    """
    Optional — same structure as TinyImageNetTest:
    Returns (image, index) for inference pipelines.
    """
    def __init__(self, transform=None, root=DATA_ROOT):
        ensure_cifar10_exists()

        self.dataset = CIFAR10(
            root=root,
            train=False,
            download=True,
        )

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]  # CIFAR-10 test has labels
        if self.transform:
            img = self.transform(img)
        return img, idx
