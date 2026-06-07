import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation_growth=2, num_layers=2, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = dilation_growth ** i
            padding = dilation * (kernel_size - 1) // 2
            layers.extend([
                nn.Conv1d(channels, channels, kernel_size, padding=padding,
                          dilation=dilation, bias=False),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultiScaleTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(15, 35, 55)):
        super().__init__()
        per_group = (out_channels // len(kernel_sizes)) * len(kernel_sizes)
        per_branch = per_group // len(kernel_sizes)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, per_branch, k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm1d(per_group)
        self.proj = nn.Conv1d(per_group, out_channels, 1, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = torch.cat([c(x) for c in self.convs], dim=1)
        out = self.bn(out)
        out = self.proj(out)
        return self.norm(out)


class StableTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=None, dropout=0.2):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop1(attn_out)
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class DSTSBlock(nn.Module):
    """Dual-Stream Temporal-Spatial block: TCN (temporal) + Transformer (spatial) summed."""
    def __init__(self, d_model=64, nhead=4, tcn_layers=3, dropout=0.25):
        super().__init__()
        self.temporal_stream = TCNBlock(d_model, num_layers=tcn_layers, dropout=dropout)
        self.spatial_stream = StableTransformerLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout
        )
        self.drop = nn.Dropout(dropout)
        self.fusion_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        t = self.temporal_stream(x)
        s = self.spatial_stream(x.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.drop(t + s)
        out = out.permute(0, 2, 1)
        out = self.fusion_norm(out)
        return out.permute(0, 2, 1)


class MTSEEGFormer(nn.Module):
    def __init__(self, n_classes=4, n_channels=22, d_model=64, dropout=0.4):
        super().__init__()
        self.temp_conv = nn.Conv2d(1, 16, (1, 25), padding=(0, 12), bias=False)
        self.bn_temp = nn.BatchNorm2d(16)
        self.spatial_conv = nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False)
        self.bn_spatial = nn.BatchNorm2d(32)

        self.down1 = nn.Sequential(
            nn.Conv2d(32, 48, (1, 7), stride=(1, 2), padding=(0, 3), bias=False),
            nn.BatchNorm2d(48), nn.GELU(), nn.Dropout(0.15),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(48, 48, (1, 7), stride=(1, 2), padding=(0, 3), bias=False),
            nn.BatchNorm2d(48), nn.GELU(), nn.Dropout(0.15),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(48, d_model, (1, 5), stride=(1, 2), padding=(0, 2), bias=False),
            nn.BatchNorm2d(d_model), nn.GELU(), nn.Dropout(0.15),
        )

        self.ms_conv = MultiScaleTemporalConv(d_model, d_model)

        self.dsts1 = DSTSBlock(d_model=d_model, nhead=4, tcn_layers=3, dropout=0.25)
        self.dsts2 = DSTSBlock(d_model=d_model, nhead=4, tcn_layers=3, dropout=0.25)
        self.drop_dsts = nn.Dropout(0.3)

        self.norm_out = RMSNorm(d_model)
        self.head_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.elu(self.bn_spatial(self.spatial_conv(F.elu(self.bn_temp(self.temp_conv(x))))))

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = x.squeeze(2)

        x = F.gelu(self.ms_conv(x))

        x = self.dsts1(x)
        x = self.dsts2(x)
        x = self.drop_dsts(x)

        x = x.mean(dim=-1)
        x = self.head_dropout(self.norm_out(x))
        return self.fc(x)
