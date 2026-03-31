import torch
import cv2
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.liteseg_edge import LiteSegEdge
from nuscenes.nuscenes import NuScenes
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── PATHS ─────────────────────────────
DATAROOT = r'C:\Users\ACER\LiteSegEdge\data\nuscenes'
VERSION  = 'v1.0-mini'
CKPT     = r'C:\Users\ACER\LiteSegEdge\checkpoints\best.pth'
OUT_VIDEO= r'C:\Users\ACER\LiteSegEdge\demo\final_demo.mp4'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── LOAD MODEL ─────────────────────────
model = LiteSegEdge().to(DEVICE)
ckpt  = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()

transform = A.Compose([
    A.Resize(360, 640),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ── LOAD DATA ──────────────────────────
nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

# ── VIDEO WRITER ───────────────────────
video = cv2.VideoWriter(
    OUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    20,
    (640, 360)
)

print("Generating video...")

# ── LOOP THROUGH FRAMES ────────────────
for i, sample in enumerate(nusc.sample[:200]):  # adjust length if needed

    cam = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    img_path = os.path.join(DATAROOT, cam['filename'])

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 360))

    tensor = transform(image=img)['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        seg, lv = model(tensor)

    prob = torch.sigmoid(seg).squeeze().cpu().numpy()
    unc  = torch.sigmoid(lv).squeeze().cpu().numpy()

    # 🔥 UNCERTAINTY FIX
    unc = cv2.GaussianBlur(unc, (7,7), 0)
    unc = np.clip(unc, 0, 0.6)

    # 🔥 MASK FIX
    mask = (prob > 0.68).astype(np.uint8)
    h, w = mask.shape
    mask[:int(h*0.45), :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))

    prob = prob * mask

    # ── COLOR OVERLAY ────────────────
    overlay = img.copy()

    green = np.zeros_like(img)
    green[:,:,1] = 255

    overlay = np.where(np.stack([mask]*3,2),
                       cv2.addWeighted(overlay, 0.6, green, 0.4, 0),
                       overlay)

    # ── RISK CALC ───────────────────
    dp = mask.mean()*100
    p75 = np.percentile(unc[mask],75) if mask.any() else 1

    cov_risk = (1 - dp/100)**2 * 40
    unc_risk = p75 * 15

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    brightness = gray.mean()
    night = 0
    if brightness < 120:
        night = (120 - brightness)/120 * 40

    score = int(min(100, cov_risk + unc_risk + night))

    if dp > 45 and p75 < 0.3:
        score -= 10

    if score >= 60:
        label, color = "HAZARD", (220,50,50)
    elif score >= 30:
        label, color = "CAUTION", (255,200,0)
    else:
        label, color = "SAFE", (0,220,80)

    # ── DRAW TEXT ───────────────────
    cv2.putText(overlay, f"{label} {score}/100",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)

    # ── WRITE FRAME ────────────────
    video.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    if i % 20 == 0:
        print(f"Frame {i} processed")

video.release()
print(f"Saved video to: {OUT_VIDEO}")