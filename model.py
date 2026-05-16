"""
Improved EEGNet with Multi-Scale Temporal Convolutions and Attention
Based on insights from MSCARNet paper - Optimized version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimAM(nn.Module):
    """Simple Attention Module - lightweight attention for EEG"""
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        # x: [B, C, H, W]
        n = x.numel() // x.size(1)
        t = x.sum(dim=[2, 3], keepdim=True)
        mean = t / n
        var = ((x - mean) ** 2).sum(dim=[2, 3], keepdim=True) / n
        e = (4 * (var + self.e_lambda)) / ((x - mean) ** 2 + 4 * var + 4 * self.e_lambda)
        return x * torch.sigmoid(e)


class MultiScaleTempConv(nn.Module):
    """Multi-scale temporal convolution block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Make channels divisible by 3
        mid_channels = (out_channels // 3) * 3
        self.actual_channels = mid_channels
        # Three kernel sizes: 15, 35, 55 (as per MSCARNet)
        self.conv1 = nn.Conv2d(in_channels, self.actual_channels // 3, (1, 15), padding=(0, 7), bias=False)
        self.conv2 = nn.Conv2d(in_channels, self.actual_channels // 3, (1, 35), padding=(0, 17), bias=False)
        self.conv3 = nn.Conv2d(in_channels, self.actual_channels // 3, (1, 55), padding=(0, 27), bias=False)
        self.bn = nn.BatchNorm2d(self.actual_channels)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return self.bn(out)


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual


class EEGNetMultiScale(nn.Module):
    """EEGNet with Multi-Scale Temporal Convolutions and Attention"""
    def __init__(self, n_classes=4, n_channels=22):
        super().__init__()

        # Block 1: Temporal conv + Depthwise
        self.conv1 = nn.Conv2d(1, 16, (1, 25), padding=(0, 12), bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Depthwise spatial filtering
        self.depthwise = nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.25)

        # Block 2: Multi-scale temporal conv (inspired by MSCARNet) - use 63 (divisible by 3)
        self.msconv = MultiScaleTempConv(32, 63)
        self.attention = SimAM()
        self.pool2 = nn.AvgPool2d((1, 4))
        self.drop2 = nn.Dropout(0.25)

        # Block 3: Additional temporal features
        self.conv3 = nn.Conv2d(63, 63, (1, 7), padding=(0, 3), bias=False)
        self.bn3 = nn.BatchNorm2d(63)
        self.pool3 = nn.AvgPool2d((1, 4))
        self.drop3 = nn.Dropout(0.25)

        # Block 4: Residual block for deeper features
        self.res_block = ResidualBlock(63)
        self.drop4 = nn.Dropout(0.25)

        # Classifier
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 4))
        self.fc = nn.Sequential(
            nn.Linear(63 * 4, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, C, T]

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2: Multi-scale + Attention
        x = self.msconv(x)
        x = self.attention(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.drop3(x)

        # Block 4: Residual
        x = self.res_block(x)
        x = self.drop4(x)

        # Classification
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class EEGNetShallow(nn.Module):
    """Shallow but effective EEGNet - simpler architecture for harder subjects"""
    def __init__(self, n_classes=4, n_channels=22):
        super().__init__()

        # Block 1: Temporal conv + Depthwise
        self.conv1 = nn.Conv2d(1, 16, (1, 25), padding=(0, 12), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.depthwise = nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.25)

        # Block 2: Separable conv
        self.separable = nn.Conv2d(32, 64, (1, 15), padding=(0, 7), bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(0.25)

        # Block 3: Deep conv
        self.conv3 = nn.Conv2d(64, 64, (1, 7), padding=(0, 3), bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool3 = nn.AvgPool2d((1, 4))
        self.drop3 = nn.Dropout(0.25)

        # Global pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 8))

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(64 * 8, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.separable(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.drop3(x)

        # Classification
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class EEGNetDeep(nn.Module):
    """Deeper EEGNet with residual connections and more capacity"""
    def __init__(self, n_classes=4, n_channels=22):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(1, 16, (1, 25), padding=(0, 12), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.depthwise = nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.25)

        # Block 2: Multi-scale
        self.msconv1 = MultiScaleTempConv(32, 63)
        self.att1 = SimAM()
        self.pool2 = nn.AvgPool2d((1, 4))
        self.drop2 = nn.Dropout(0.25)

        # Block 3
        self.conv3 = nn.Conv2d(63, 63, (1, 7), padding=(0, 3), bias=False)
        self.bn3 = nn.BatchNorm2d(63)
        self.pool3 = nn.AvgPool2d((1, 4))
        self.drop3 = nn.Dropout(0.25)

        # Block 4: Residual
        self.res1 = ResidualBlock(63)
        self.drop4 = nn.Dropout(0.25)

        # Block 5: Additional conv
        self.conv4 = nn.Conv2d(63, 64, (1, 3), padding=(0, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.drop5 = nn.Dropout(0.25)

        # Global pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 2))

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.msconv1(x)
        x = self.att1(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.drop3(x)

        # Block 4
        x = self.res1(x)
        x = self.drop4(x)

        # Block 5
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.drop5(x)

        # Classification
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Default to the best model
BestEEGNet = EEGNetMultiScale