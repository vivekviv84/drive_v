import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────
# 1. Focal Loss  (handles class imbalance)
# ─────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # logits: B x 1 x H x W,  targets: B x H x W (int64)
        targets_f = targets.float().unsqueeze(1)          # B x 1 x H x W
        bce       = F.binary_cross_entropy_with_logits(
                        logits, targets_f, reduction='none')
        pt        = torch.exp(-bce)
        focal     = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ─────────────────────────────────────────
# 2. Lovász-Softmax Loss  (optimises mIoU)
# ─────────────────────────────────────────
def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszBinaryLoss(nn.Module):
    def forward(self, logits, targets):
        # logits: B x 1 x H x W,  targets: B x H x W
        logits  = logits.squeeze(1)                       # B x H x W
        loss    = 0.0
        batch   = logits.shape[0]
        for i in range(batch):
            log_i = logits[i].view(-1)
            tgt_i = targets[i].view(-1).float()
            signs = 2.0 * tgt_i - 1.0
            errors = 1.0 - log_i * signs
            errors_sorted, perm = torch.sort(errors, descending=True)
            gt_sorted = tgt_i[perm]
            grad = lovasz_grad(gt_sorted)
            loss += torch.dot(F.relu(errors_sorted), grad)
        return loss / batch
    

class BoundaryLoss(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, logits, targets):
        # extract boundaries from target mask
        targets_f  = targets.float().unsqueeze(1)   # B x 1 x H x W
        kernel     = torch.ones(1, 1, self.kernel_size,
                                self.kernel_size,
                                device=targets.device)
        pad        = self.kernel_size // 2
        dilated    = torch.clamp(
            torch.nn.functional.conv2d(
                targets_f, kernel, padding=pad), 0, 1)
        eroded     = 1 - torch.clamp(
            torch.nn.functional.conv2d(
                1 - targets_f, kernel, padding=pad), 0, 1)
        boundary   = (dilated - eroded)             # B x 1 x H x W

        # weight BCE loss by boundary region
        bce        = F.binary_cross_entropy_with_logits(
                         logits, targets_f, reduction='none')
        boundary_loss = (bce * (1 + 5 * boundary)).mean()
        return boundary_loss


# ─────────────────────────────────────────
# 3. Uncertainty Regulariser
#    Heteroscedastic loss — learns per-pixel
#    aleatoric uncertainty (log variance)
# ─────────────────────────────────────────
class UncertaintyLoss(nn.Module):
    def forward(self, logits, log_var, targets):
        # log_var: B x 1 x H x W
        targets_f  = targets.float().unsqueeze(1)
        precision  = torch.exp(-log_var)               # 1/sigma^2
        bce        = F.binary_cross_entropy_with_logits(
                         logits, targets_f, reduction='none')
        unc_loss   = (precision * bce + 0.5 * log_var).mean()
        return unc_loss


# ─────────────────────────────────────────
# 4. Combined Total Loss
# ─────────────────────────────────────────
class TotalLoss(nn.Module):
    def __init__(self, lambda_focal=1.0,
                       lambda_lovasz=0.5,
                       lambda_unc=0.3,
                       lambda_boundary=0.5):
        super().__init__()
        self.focal    = FocalLoss()
        self.lovasz   = LovaszBinaryLoss()
        self.unc      = UncertaintyLoss()
        self.boundary = BoundaryLoss()
        self.lf       = lambda_focal
        self.ll       = lambda_lovasz
        self.lu       = lambda_unc
        self.lb       = lambda_boundary

    def forward(self, logits, log_var, targets):
        l_focal    = self.focal(logits, targets)
        l_lovasz   = self.lovasz(logits, targets)
        l_unc      = self.unc(logits, log_var, targets)
        l_boundary = self.boundary(logits, targets)

        total = (self.lf * l_focal    +
                 self.ll * l_lovasz   +
                 self.lu * l_unc      +
                 self.lb * l_boundary)

        return total, {
            'focal'    : l_focal.item(),
            'lovasz'   : l_lovasz.item(),
            'unc'      : l_unc.item(),
            'boundary' : l_boundary.item(),
            'total'    : total.item(),
        }

# ─────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────
if __name__ == '__main__':
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = TotalLoss().to(device)

    logits  = torch.randn(4, 1, 360, 640).to(device)
    log_var = torch.randn(4, 1, 360, 640).to(device)
    targets = torch.randint(0, 2, (4, 360, 640)).to(device)

    total, breakdown = loss_fn(logits, log_var, targets)

    print(f"Device         : {device}")
    print(f"Focal loss     : {breakdown['focal']:.4f}")
    print(f"Lovasz loss    : {breakdown['lovasz']:.4f}")
    print(f"Uncertainty    : {breakdown['unc']:.4f}")
    print(f"Total loss     : {breakdown['total']:.4f}")
    print("\nLoss functions ready!")