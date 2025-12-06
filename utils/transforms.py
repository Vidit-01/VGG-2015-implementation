from torchvision import transforms



val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
])
