import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Squeeze-and-Excite block ----------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# ---------- DW-Asymmetric stage ----------
class DWAsymStage(nn.Module):
    """
    Depthwise asymmetric conv: 1x7 then 7x1
    + pointwise projection
    + Squeeze-Excite gate
    + MaxPool2d downsample
    """
    def __init__(self, in_ch, out_ch, reduction=8):
        super().__init__()
        # depthwise asymmetric
        self.dw_h = nn.Conv2d(in_ch, in_ch, (1,7), padding=(0,3), groups=in_ch)
        self.dw_v = nn.Conv2d(in_ch, in_ch, (7,1), padding=(3,0), groups=in_ch)
        # pointwise expand
        self.pw   = nn.Conv2d(in_ch, out_ch, 1)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.se   = SEBlock(out_ch, reduction)
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.pw(self.dw_v(self.dw_h(x))))
        x = self.bn(x)
        x = self.se(x)
        skip = x                  # save before downsample
        x    = self.down(x)
        return x, skip            # (downsampled, skip connection)


# ---------- Full Encoder ----------
class LiteEncoder(nn.Module):
    """
    3 DW-Asym stages: 3 -> 16 -> 32 -> 64 channels
    Input : B x 3  x 360 x 640
    Output: B x 64 x 45  x 80   +  3 skip tensors
    """
    def __init__(self):
        super().__init__()
        self.stage1 = DWAsymStage(3,  16)
        self.stage2 = DWAsymStage(16, 32)
        self.stage3 = DWAsymStage(32, 64)

    def forward(self, x):
        x,  s1 = self.stage1(x)   # 16 x 180 x 320
        x,  s2 = self.stage2(x)   # 32 x  90 x 160
        x,  s3 = self.stage3(x)   # 64 x  45 x  80
        return x, [s1, s2, s3]


# ---------- quick test ----------
if __name__ == '__main__':
    model = LiteEncoder()
    total = sum(p.numel() for p in model.parameters())
    print(f"Encoder params: {total:,}")

    dummy = torch.randn(2, 3, 360, 640)
    out, skips = model(dummy)
    print(f"Encoder output : {out.shape}")
    for i, s in enumerate(skips):
        print(f"  skip[{i+1}]      : {s.shape}")