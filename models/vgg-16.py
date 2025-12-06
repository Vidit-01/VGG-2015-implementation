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
            nn.MaxPool2d(2, 2),

            # Block 2
            conv_bn(64, 128),
            conv_bn(128, 128),
            nn.MaxPool2d(2, 2),

            # Block 3
            conv_bn(128, 256),
            conv_bn(256, 256),
            conv_bn(256, 256),
            nn.MaxPool2d(2, 2),

            # Block 4 - Reduced to 384 instead of 512
            conv_bn(256, 384),
            conv_bn(384, 384),
            conv_bn(384, 384),
            nn.MaxPool2d(2, 2),

            # Block 5 - Reduced to 384 instead of 512
            conv_bn(384, 384),
            conv_bn(384, 384),
            conv_bn(384, 384),
            nn.MaxPool2d(2, 2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        # Much smaller classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(384 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    

import torch.nn as nn
model = VGG()
# Assuming 'model' is your PyTorch nn.Module instance
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
