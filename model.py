import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedEEGNet(nn.Module):
    def __init__(self, n_classes=4, n_channels=22):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding=(0, 31), bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.depthwise = nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.35)

        self.conv2 = nn.Conv2d(16, 32, (1, 16), padding=(0, 7), bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(0.35)

        self.conv3 = nn.Conv2d(32, 32, (1, 8), padding=(0, 3), bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.pool3 = nn.AvgPool2d((1, 4))
        self.drop3 = nn.Dropout(0.35)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 8))
        self.fc = nn.Sequential(
            nn.Linear(32 * 8, 64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
