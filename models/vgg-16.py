import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()

        def conv_bn(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.features = nn.Sequential(
            # Block 1
            conv_bn(3, 64),
            conv_bn(64, 64),
            nn.MaxPool2d(2),

            # Block 2
            conv_bn(64, 128),
            conv_bn(128, 128),
            nn.MaxPool2d(2),

            # Block 3
            conv_bn(128, 256),
            conv_bn(256, 256),
            conv_bn(256, 256),
            nn.MaxPool2d(2),

            # Block 4
            conv_bn(256, 512),
            conv_bn(512, 512),
            conv_bn(512, 512),
            nn.MaxPool2d(2),

            # Block 5
            conv_bn(512, 512),
            conv_bn(512, 512),
            conv_bn(512, 512),
            nn.MaxPool2d(2),
        )

        # For TinyImageNet (64x64): after 5 pools -> 2x2
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
