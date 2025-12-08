import os
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010))
])

trainset = CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
testset = CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

