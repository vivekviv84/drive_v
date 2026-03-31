import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.liteseg_edge import LiteSegEdge
from dataset.nuscenes_dataset import NuScenesSegDataset, get_train_transforms, get_val_transforms
from losses.losses import TotalLoss

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
MASK_DIR    = r'C:\Users\ACER\LiteSegEdge\data\masks'
CKPT_DIR    = r'C:\Users\ACER\LiteSegEdge\checkpoints'
RUNS_DIR    = r'C:\Users\ACER\LiteSegEdge\runs'
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

EPOCHS      = 60
BATCH_SIZE  = 8
LR          = 3e-4
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3-phase curriculum
# Phase 1 (ep  1-20): focal only
# Phase 2 (ep 21-40): focal + lovasz
# Phase 3 (ep 41-60): focal + lovasz + uncertainty
def get_loss_weights(epoch):
    if epoch <= 20:
        return dict(lambda_focal=1.0, lambda_lovasz=0.0,
                    lambda_unc=0.0,   lambda_boundary=0.3)
    elif epoch <= 40:
        return dict(lambda_focal=1.0, lambda_lovasz=0.5,
                    lambda_unc=0.0,   lambda_boundary=0.5)
    else:
        return dict(lambda_focal=1.0, lambda_lovasz=0.5,
                    lambda_unc=0.3,   lambda_boundary=0.5)

# ─────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────
def compute_miou(logits, targets, threshold=0.5):
    preds   = (torch.sigmoid(logits.squeeze(1)) > threshold).long()
    inter   = (preds & targets).sum(dim=(1,2)).float()
    union   = (preds | targets).sum(dim=(1,2)).float()
    iou     = (inter + 1e-6) / (union + 1e-6)
    return iou.mean().item()


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")

    # datasets
    train_ds = NuScenesSegDataset(MASK_DIR, split='train',
                                  transform=get_train_transforms())
    val_ds   = NuScenesSegDataset(MASK_DIR, split='val',
                                  transform=get_val_transforms())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    # model
    model   = LiteSegEdge().to(DEVICE)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_p:,}")

    # optimizer + scheduler + scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler()

    best_miou   = 0.0
    history     = {'train_loss': [], 'val_loss': [], 'val_miou': []}

    for epoch in range(1, EPOCHS + 1):
        # update loss curriculum
        weights  = get_loss_weights(epoch)
        loss_fn  = TotalLoss(**weights).to(DEVICE)

        # ── train ──
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader,
                    desc=f"Ep {epoch:02d}/{EPOCHS} [train]",
                    leave=False)
        for imgs, masks in pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                logits, log_var      = model(imgs)
                loss, _              = loss_fn(logits, log_var, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss /= len(train_loader)

        # ── validate ──
        model.eval()
        val_loss = 0.0
        val_miou = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks          = imgs.to(DEVICE), masks.to(DEVICE)
                with autocast():
                    logits, log_var  = model(imgs)
                    loss, _          = loss_fn(logits, log_var, masks)
                val_loss += loss.item()
                val_miou += compute_miou(logits, masks)

        val_loss /= len(val_loader)
        val_miou /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_miou)

        print(f"Ep {epoch:02d}/{EPOCHS} | "
              f"train={train_loss:.4f} | "
              f"val={val_loss:.4f} | "
              f"mIoU={val_miou:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

        # save best
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                'epoch'     : epoch,
                'model'     : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'miou'      : best_miou,
            }, os.path.join(CKPT_DIR, 'best.pth'))
            print(f"  ✓ Best model saved  mIoU={best_miou:.4f}")

        # save every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(CKPT_DIR, f'epoch_{epoch:03d}.pth'))

    # ── plot loss curves ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'],   label='Val Loss')
    ax1.set_title('Loss Curves'); ax1.legend(); ax1.set_xlabel('Epoch')
    ax2.plot(history['val_miou'],   label='Val mIoU', color='green')
    ax2.set_title('Validation mIoU'); ax2.legend(); ax2.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(RUNS_DIR, 'training_curves.png'), dpi=150)
    print(f"\nTraining complete! Best mIoU: {best_miou:.4f}")
    print(f"Curves saved to runs/training_curves.png")


if __name__ == '__main__':
    main()