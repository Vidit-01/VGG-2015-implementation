import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms


train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=val_transform
)



