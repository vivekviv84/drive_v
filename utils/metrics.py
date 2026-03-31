import torch
import torch.nn.functional as F
import numpy as np
import cv2
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────
# 1. mIoU
# ─────────────────────────────────────────
def compute_miou(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits.squeeze(1)) > threshold).long()
    inter = (preds & targets).sum(dim=(1,2)).float()
    union = (preds | targets).sum(dim=(1,2)).float()
    iou   = (inter + 1e-6) / (union + 1e-6)
    return iou.mean().item()


# ─────────────────────────────────────────
# 2. Boundary F1
# ─────────────────────────────────────────
def mask_to_boundary(mask_np, dilation=3):
    """Extract boundary pixels from a binary mask."""
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation, dilation))
    dilated  = cv2.dilate(mask_np, kernel)
    boundary = dilated - mask_np
    return boundary

def compute_boundary_f1(logits, targets, threshold=0.5, dilation=11):
    preds   = (torch.sigmoid(logits.squeeze(1)) > threshold).long()
    preds   = preds.cpu().numpy().astype(np.uint8)
    targets = targets.cpu().numpy().astype(np.uint8)

    f1_scores = []
    for pred, tgt in zip(preds, targets):
        pred_b = mask_to_boundary(pred, dilation)
        tgt_b  = mask_to_boundary(tgt,  dilation)

        tp = (pred_b & tgt_b).sum()
        fp = (pred_b & (1 - tgt_b)).sum()
        fn = ((1 - pred_b) & tgt_b).sum()

        precision = tp / (tp + fp + 1e-6)
        recall    = tp / (tp + fn + 1e-6)
        f1        = 2 * precision * recall / (precision + recall + 1e-6)
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


# ─────────────────────────────────────────
# 3. Full Evaluation Script
# ─────────────────────────────────────────
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from models.liteseg_edge import LiteSegEdge
    from dataset.nuscenes_dataset import NuScenesSegDataset, get_val_transforms

    MASK_DIR = r'C:\Users\ACER\LiteSegEdge\data\masks'
    CKPT     = r'C:\Users\ACER\LiteSegEdge\checkpoints\best.pth'
    DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model    = LiteSegEdge().to(DEVICE)
    ckpt     = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"(best mIoU={ckpt['miou']:.4f})")

    # dataloader
    val_ds  = NuScenesSegDataset(MASK_DIR, split='val',
                                 transform=get_val_transforms())
    loader  = DataLoader(val_ds, batch_size=4,
                         shuffle=False, num_workers=0)

    all_miou = []
    all_bf1  = []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits, _   = model(imgs)

            all_miou.append(compute_miou(logits, masks))
            all_bf1.append(compute_boundary_f1(logits, masks))

    miou = np.mean(all_miou)
    bf1  = np.mean(all_bf1)

    print(f"\n{'='*40}")
    print(f"  Final Validation Results")
    print(f"{'='*40}")
    print(f"  mIoU         : {miou:.4f}  ({miou*100:.2f}%)")
    print(f"  Boundary F1  : {bf1:.4f}  ({bf1*100:.2f}%)")
    print(f"{'='*40}")
    print(f"\nTarget mIoU >72% : {'✓ PASSED' if miou > 0.72 else '✗ needs more training'}")