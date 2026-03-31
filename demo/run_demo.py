import torch
import cv2
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.liteseg_edge import LiteSegEdge
from nuscenes.nuscenes import NuScenes
import albumentations as A
from albumentations.pytorch import ToTensorV2

DATAROOT = r'C:\Users\ACER\LiteSegEdge\data\nuscenes'
VERSION  = 'v1.0-mini'
CKPT     = r'C:\Users\ACER\LiteSegEdge\checkpoints\best.pth'
OUT_DIR  = r'C:\Users\ACER\LiteSegEdge\demo'
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── load model ──
model = LiteSegEdge().to(DEVICE)
ckpt  = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"Model loaded  — {sum(p.numel() for p in model.parameters()):,} params")

# ── transforms ──
transform = A.Compose([
    A.Resize(360, 640),
    A.Normalize(mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

# ── video writer ──
video_path = os.path.join(OUT_DIR, 'demo_video.mp4')
writer     = cv2.VideoWriter(
    video_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    10, (640*3, 360)   # 3 panels side by side
)

print(f"Generating demo video from {len(nusc.sample)} frames...")

for i, sample in enumerate(nusc.sample):
    cam_token = sample['data']['CAM_FRONT']
    cam_data  = nusc.get('sample_data', cam_token)
    img_path  = os.path.join(DATAROOT, cam_data['filename'])

    img_bgr   = cv2.imread(img_path)
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # preprocess
    aug       = transform(image=img_rgb)
    tensor    = aug['image'].unsqueeze(0).to(DEVICE)

    # inference
    with torch.no_grad():
        seg_logits, log_var = model(tensor)

    # post-process
    seg_prob  = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
    unc_map   = torch.sigmoid(log_var).squeeze().cpu().numpy()
    seg_mask  = (seg_prob > 0.5).astype(np.uint8)

    # resize outputs to match display
    img_disp  = cv2.resize(img_bgr, (640, 360))

    # panel 1: original
    panel1    = img_disp.copy()
    cv2.putText(panel1, 'Input', (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # panel 2: overlay (green = drivable)
    overlay   = img_disp.copy()
    green     = np.zeros_like(img_disp)
    green[:,:,1] = 255
    mask_3ch  = np.stack([seg_mask]*3, axis=2)
    overlay   = np.where(mask_3ch, 
                         cv2.addWeighted(img_disp, 0.5, green, 0.5, 0),
                         img_disp)
    # draw boundary
    contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    cv2.putText(overlay, f'Drivable  mIoU=72.75%', (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # panel 3: uncertainty heatmap
    unc_norm  = (unc_map * 255).astype(np.uint8)
    unc_color = cv2.applyColorMap(unc_norm, cv2.COLORMAP_JET)
    cv2.putText(unc_color, 'Uncertainty', (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # add FPS text
    cv2.putText(overlay, '287 FPS', (10, 360-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # combine panels
    frame     = np.concatenate([panel1, overlay, unc_color], axis=1)
    writer.write(frame)

    if i % 50 == 0:
        print(f"  [{i+1}/{len(nusc.sample)}] frames written")

writer.release()
print(f"\nDemo video saved to {video_path}")
print("Step 12 complete — PROJECT DONE!")