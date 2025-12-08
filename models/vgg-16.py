import torch
import torch.nn as nn

<<<<<<< HEAD
class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
=======

class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        def conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            )

        self.features = nn.Sequential(
            # Block 1
            conv(3, 64),
            conv(64, 64),
            nn.MaxPool2d(2),

            # Block 2
            conv(64, 128),
            conv(128, 128),
            nn.MaxPool2d(2),

            # Block 3
            conv(128, 256),
            conv(256, 256),
            conv(256, 256),
            nn.MaxPool2d(2),

            # Block 4 (reduced)
            conv(256, 256),
            conv(256, 256),
            conv(256, 256),
            nn.MaxPool2d(2),

            # Block 5 (reduced)
            conv(256, 256),
            conv(256, 256),
            conv(256, 256),
            nn.MaxPool2d(2),
        )

        # Output is already 2×2 — no need for adaptive pool
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(12544, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
>>>>>>> 10052d895ca3f04c683efb6741563810c58b89ff
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
<<<<<<< HEAD
        return self.classifier(x)
=======
        x = self.classifier(x)
        return x

if __name__=="__main__":
    model = VGG()
    model.forward(torch.randn(1,3,64,64))
>>>>>>> 10052d895ca3f04c683efb6741563810c58b89ff
