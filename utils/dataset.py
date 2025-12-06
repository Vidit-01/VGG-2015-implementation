import os
import urllib.request
import zipfile
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
import shutil

VAL_ANNOTATION_FILE = "val_annotations.txt"
DATA_ROOT = "data/tinyimagenet"
DOWNLOAD_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


# ------------------------------------------------------
# Auto-download + extract if dataset is missing
# ------------------------------------------------------
def ensure_tinyimagenet_exists():
    if os.path.exists(DATA_ROOT):
        return  # already present

    os.makedirs("data", exist_ok=True)
    zip_path = "data/tiny-imagenet-200.zip"

    print("⬇️ Downloading TinyImageNet (≈ 240MB)...")
    urllib.request.urlretrieve(DOWNLOAD_URL, zip_path)

    print("📦 Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("data")

    os.remove(zip_path)

    # Rename folder to expected name if needed
    if os.path.exists("data/tiny-imagenet-200") and not os.path.exists(DATA_ROOT):
        shutil.move("data/tiny-imagenet-200", DATA_ROOT)

    print("✅ TinyImageNet ready at", DATA_ROOT)


# ------------------------------------------------------
# Dataset classes
# ------------------------------------------------------
class TinyImageNetTrain(ImageFolder):
    def __init__(self, transform=None, root=f"{DATA_ROOT}/train"):
        ensure_tinyimagenet_exists()
        super().__init__(root=root, transform=transform)


class TinyImageNetVal(Dataset):
    def __init__(self, transform=None, root=f"{DATA_ROOT}/val"):
        ensure_tinyimagenet_exists()
        self.root = root
        self.transform = transform

        annot_path = os.path.join(root, VAL_ANNOTATION_FILE)

        # read image-label pairs
        self.samples = []
        with open(annot_path, "r") as f:
            for line in f:
                img, cls, *_ = line.strip().split("\t")
                img_path = os.path.join(root, "images", img)
                self.samples.append((img_path, cls))

        # build class-to-index mapping
        classes = sorted(list(set(cls for _, cls in self.samples)))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        # convert class string to numeric label
        self.samples = [(img, self.class_to_idx[cls]) for img, cls in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


class TinyImageNetTest(Dataset):
    def __init__(self, transform=None, root=f"{DATA_ROOT}/test/images"):
        ensure_tinyimagenet_exists()
        self.root = root
        self.transform = transform
        self.files = sorted(os.listdir(root))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.root, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, img_name
