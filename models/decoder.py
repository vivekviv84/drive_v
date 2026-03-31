import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import LiteEncoder


# ---------- DW-Sep conv refine block ----------
class DWSepConv(nn.Module):
    """Depthwise-separable conv + BN + ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch,  3, padding=1, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))))


# ---------- Decoder ----------
class LiteDecoder(nn.Module):
    """
    3x bilinear upsample stages with skip fusion.
    Encoder skips : [s1=16ch, s2=32ch, s3=64ch]
    Bottleneck in : 64ch
    """
    def __init__(self):
        super().__init__()

        # after concat(64 + 64) = 128 -> refine to 64
        self.refine3 = DWSepConv(128, 64)
        # after concat(64 + 32) = 96  -> refine to 32
        self.refine2 = DWSepConv(96,  32)
        # after concat(32 + 16) = 48  -> refine to 16
        self.refine1 = DWSepConv(48,  16)

    def forward(self, x, skips):
        s1, s2, s3 = skips          # s1=largest, s3=smallest

        # stage 3: upsample x (64) -> match s3 (64) -> concat -> refine
        x = F.interpolate(x, size=s3.shape[2:], mode='bilinear', align_corners=False)
        x = self.refine3(torch.cat([x, s3], dim=1))   # 128 -> 64

        # stage 2: upsample -> match s2 -> concat -> refine
        x = F.interpolate(x, size=s2.shape[2:], mode='bilinear', align_corners=False)
        x = self.refine2(torch.cat([x, s2], dim=1))   # 96 -> 32

        # stage 1: upsample -> match s1 -> concat -> refine
        x = F.interpolate(x, size=s1.shape[2:], mode='bilinear', align_corners=False)
        x = self.refine1(torch.cat([x, s1], dim=1))   # 48 -> 16

        return x   # B x 16 x 360 x 640


# ---------- quick test ----------
if __name__ == '__main__':
    from encoder import LiteEncoder

    enc   = LiteEncoder()
    dec   = LiteDecoder()

    total = sum(p.numel() for p in enc.parameters()) + \
            sum(p.numel() for p in dec.parameters())
    print(f"Encoder + Decoder params: {total:,}")

    dummy       = torch.randn(2, 3, 360, 640)
    bottleneck, skips = enc(dummy)
    out         = dec(bottleneck, skips)

    print(f"Decoder output : {out.shape}")
    assert out.shape == (2, 16, 360, 640), "Shape mismatch!"
    print("Decoder test passed!")