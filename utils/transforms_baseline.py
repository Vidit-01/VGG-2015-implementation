from torchvision import transforms


train_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
])





