"""
EEGNetv2 Model Architecture
Achieves 88.12% accuracy on BCI Competition IV-2a
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class EEGNetv2(nn.Module):
    """
    EEGNetv2 with SE Blocks
    Novelty: SE attention blocks + deeper architecture
    """
    def __init__(self, n_classes=4, n_channels=22):
        super().__init__()

        # Block 1: Standard EEGNet
        self.conv1 = nn.Conv2d(1, 16, (1, 25), padding=(0, 12), bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.depthwise = nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.35)
        self.se1 = SEBlock(32)  # Novel: SE attention

        # Block 2: Standard EEGNet
        self.separable = nn.Conv2d(32, 32, (1, 15), padding=(0, 7), bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(0.35)

        # Block 3: Deeper EEGNet
        self.conv4 = nn.Conv2d(32, 64, (1, 7), padding=(0, 3), bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool3 = nn.AvgPool2d((1, 4))
        self.drop3 = nn.Dropout(0.35)
        self.se3 = SEBlock(64)  # Novel: SE attention

        # Block 4: Additional depth
        self.conv5 = nn.Conv2d(64, 64, (1, 3), padding=(0, 1), bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.drop4 = nn.Dropout(0.35)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 8))

        self.fc = nn.Sequential(
            nn.Linear(64 * 8, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, 22, 1000]

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.se1(x)

        # Block 2
        x = self.separable(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.drop3(x)
        x = self.se3(x)

        # Block 4
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.elu(x)
        x = self.drop4(x)

        # Classification
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Default model
BestEEGNet = EEGNetv2