import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import LiteEncoder
from decoder import LiteDecoder


class LiteSegEdge(nn.Module):
    """
    Full model:
      Encoder  -> bottleneck features + skip connections
      Decoder  -> full-res feature map (16ch)
      Seg head -> 1x1 conv -> binary mask  (B x 1 x H x W)
      Unc head -> 1x1 conv -> sigma^2 map  (B x 1 x H x W)
    """
    def __init__(self):
        super().__init__()
        self.encoder  = LiteEncoder()
        self.decoder  = LiteDecoder()

        # segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
        )

        # uncertainty head (predicts log variance for stability)
        self.unc_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        features          = self.decoder(bottleneck, skips)

        seg_logits = self.seg_head(features)   # raw logits  B x 1 x H x W
        log_var    = self.unc_head(features)   # log sigma^2 B x 1 x H x W

        return seg_logits, log_var


# ---------- quick test ----------
if __name__ == '__main__':
    model  = LiteSegEdge()
    total  = sum(p.numel() for p in model.parameters())
    print(f"Total params : {total:,}")
    print(f"Target       : <2,000,000")
    assert total < 2_000_000, "Model too large!"

    dummy              = torch.randn(2, 3, 360, 640)
    seg_logits, log_var = model(dummy)

    print(f"Seg logits   : {seg_logits.shape}")
    print(f"Uncertainty  : {log_var.shape}")

    # test GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice       : {device}")
    model  = model.to(device)
    dummy  = dummy.to(device)
    seg_logits, log_var = model(dummy)
    print(f"GPU forward pass OK!")
    print(f"\nLiteSegEdge model ready!")