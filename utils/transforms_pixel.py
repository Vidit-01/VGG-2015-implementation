from torchvision.transforms import v2 as T
import torch

train_transform = T.Compose([
    T.Resize(256),
    T.RandomCrop(227),
    T.RandomHorizontalFlip(),

    # === Pixel intensity augmentations on GPU ===
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    T.RandomGrayscale(p=0.05),

    # === Convert to Tensor + move aug to GPU ===
    T.ToImage(),  # converts PIL → tensor
    T.ToDtype(torch.float32, scale=True),
])

val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(227),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
])
