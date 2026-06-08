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


# --------------------------------------------------------------------------
# Novel: learnable alignment refinement on top of (fixed, label-free) EA.
# EA whitens by a single session-level covariance reference; these modules add
# a learned spatial alignment that the backbone trains end-to-end. Both are
# identity-initialised (zero-init residual) so they cannot degrade the EA start.
# --------------------------------------------------------------------------
class SpatialAlign(nn.Module):
    """Static learnable spatial alignment: x' = (I + dW) x, dW init 0.
    Weight decay pulls dW toward 0 (i.e. toward the identity / pure EA)."""
    def __init__(self, n_channels=22):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(n_channels, n_channels))
        self.register_buffer('eye', torch.eye(n_channels))

    def forward(self, x):  # x: [B, C, T]
        W = self.eye + self.delta
        return torch.einsum('cd,bdt->bct', W, x)


class AdaptiveSpatialAlign(nn.Module):
    """Input-adaptive (per-trial) spatial alignment — the novel contribution.

    EA aligns at the SESSION level (one covariance reference per session). This
    layer predicts a per-TRIAL low-rank refinement W_b = I + alpha * U_b V_b^T from
    the trial's own per-channel log-variance, applied as x'_b = W_b x_b. The output
    head is zero-initialised so it starts exactly at the EA identity and learns
    trial-specific alignment end-to-end with the classifier (label-free at test:
    only uses the trial's own signal statistics)."""
    def __init__(self, n_channels=22, rank=4, hidden=32, alpha=0.1):
        super().__init__()
        self.C, self.r, self.alpha = n_channels, rank, alpha
        self.net = nn.Sequential(
            nn.Linear(n_channels, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * n_channels * rank),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('eye', torch.eye(n_channels))

    def forward(self, x):  # x: [B, C, T]
        v = torch.log(x.var(dim=2) + 1e-6)              # [B, C]
        uv = self.net(v).view(x.size(0), self.C, 2 * self.r)
        U, V = uv[..., :self.r], uv[..., self.r:]        # [B, C, r]
        dW = torch.bmm(U, V.transpose(1, 2))             # [B, C, C]
        W = self.eye.unsqueeze(0) + self.alpha * dW
        return torch.bmm(W, x)


class AlignedEEGNet(nn.Module):
    """UnifiedEEGNet + static learnable spatial alignment front-end."""
    def __init__(self, n_classes=4, n_channels=22):
        super().__init__()
        self.align = SpatialAlign(n_channels)
        self.backbone = UnifiedEEGNet(n_classes, n_channels)

    def forward(self, x):
        return self.backbone(self.align(x))


class AdaptiveAlignEEGNet(nn.Module):
    """UnifiedEEGNet + per-trial input-adaptive spatial alignment front-end."""
    def __init__(self, n_classes=4, n_channels=22):
        super().__init__()
        self.align = AdaptiveSpatialAlign(n_channels)
        self.backbone = UnifiedEEGNet(n_classes, n_channels)

    def forward(self, x):
        return self.backbone(self.align(x))

