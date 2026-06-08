import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqAttn(nn.Module):
    def __init__(self, n_chans=22, n_freq_chan=8, kernel_sizes=(7, 15, 31)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(1, n_freq_chan, (1, k), padding=(0, k // 2), bias=False)
            for k in kernel_sizes
        ])
        self.spatial = nn.Conv2d(n_freq_chan, 16, (n_chans, 1), groups=n_freq_chan, bias=False)
        self.bn_spat = nn.BatchNorm2d(16)
        self.n_freq_chan = n_freq_chan
        self.n_bands = len(kernel_sizes)
        attn_dim = n_freq_chan * self.n_bands
        self.freq_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(attn_dim, max(attn_dim // 4, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(attn_dim // 4, 4), attn_dim),
        )
        self.proj = nn.Conv2d(16, 16, (1, 1), bias=False)

    def forward(self, x):
        B = x.shape[0]
        bands = []
        for branch in self.branches:
            bands.append(branch(x))
        bands = torch.stack(bands, dim=1)
        attn_logits = self.freq_attn(bands)
        attn_weights = F.softmax(attn_logits.view(B, self.n_bands, self.n_freq_chan, 1, 1), dim=1)
        fused = (bands * attn_weights).sum(dim=1)
        fused = self.spatial(fused)
        fused = F.elu(self.bn_spat(fused))
        fused = self.proj(fused)
        return fused


class FreqAwareEEGNet(nn.Module):
    def __init__(self, n_classes=4, n_channels=22):
        super().__init__()
        self.stem_temp = nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False)
        self.stem_bn_t = nn.BatchNorm2d(8)
        self.stem_spat = nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False)
        self.stem_bn_s = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.35)

        self.freq_attn = FreqAttn(n_chans=n_channels, n_freq_chan=16,
                                  kernel_sizes=(7, 15, 31))

        self.gate = nn.Linear(16, 16)
        self.proj_fuse = nn.Conv2d(16, 16, (1, 1), bias=False)
        self.bn_fuse = nn.BatchNorm2d(16)

        self.block2 = nn.Conv2d(16, 32, (1, 16), padding=(0, 7), bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(0.35)

        self.block3 = nn.Conv2d(32, 32, (1, 8), padding=(0, 3), bias=False)
        self.bn3 = nn.BatchNorm2d(32)
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
        x_orig = x

        x = self.stem_temp(x)
        x = self.stem_bn_t(x)
        x = self.stem_spat(x)
        x = self.stem_bn_s(x)
        x = F.elu(x)

        freq = self.freq_attn(x_orig)
        x = self.pool1(x)
        freq = self.pool1(freq)
        gate = torch.sigmoid(self.gate(x.mean(dim=[2, 3]))).view(-1, 16, 1, 1)
        x = x + gate * freq

        x = self.drop1(x)
        x = F.elu(self.bn_fuse(self.proj_fuse(x)))

        x = F.elu(self.bn2(self.block2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.elu(self.bn3(self.block3(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

