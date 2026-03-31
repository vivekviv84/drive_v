import torch
import cv2
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# load model
model = LiteSegEdge().to(DEVICE)
ckpt  = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()

transform = A.Compose([
    A.Resize(360, 640),
    A.Normalize(mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

def run_inference(img_rgb):
    aug     = transform(image=img_rgb)
    tensor  = aug['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        seg_logits, log_var = model(tensor)
    prob    = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
    unc     = torch.sigmoid(log_var).squeeze().cpu().numpy()
    mask    = (prob > 0.5).astype(np.uint8)
    return prob, unc, mask

def make_overlay(img_bgr, mask, unc):
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    overlay  = img_rgb.copy()
    green    = np.zeros_like(img_rgb)
    green[:,:,1] = 255
    mask_3ch = np.stack([mask]*3, axis=2)
    overlay  = np.where(mask_3ch,
                        cv2.addWeighted(img_rgb, 0.45, green, 0.55, 0),
                        img_rgb)
    # cyan boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    # high uncertainty regions in red
    high_unc = (unc > 0.55).astype(np.uint8)
    red      = np.zeros_like(img_rgb)
    red[:,:,0] = 255
    unc_3ch  = np.stack([high_unc]*3, axis=2)
    overlay  = np.where(unc_3ch,
                        cv2.addWeighted(overlay, 0.5, red, 0.5, 0),
                        overlay)
    return overlay

# pick 6 diverse frames
indices  = [0, 40, 80, 120, 200, 300]
fig      = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor('#0d1117')
gs       = gridspec.GridSpec(3, 6, figure=fig,
                              hspace=0.35, wspace=0.05)

labels   = ['Frame 1', 'Frame 2', 'Frame 3',
            'Frame 4', 'Frame 5', 'Frame 6']

print("Generating edge case demo...")
for col, (idx, label) in enumerate(zip(indices, labels)):
    sample    = nusc.sample[idx]
    cam_token = sample['data']['CAM_FRONT']
    cam_data  = nusc.get('sample_data', cam_token)
    img_path  = os.path.join(DATAROOT, cam_data['filename'])
    img_bgr   = cv2.imread(img_path)
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb   = cv2.resize(img_rgb, (640, 360))
    img_bgr   = cv2.resize(img_bgr, (640, 360))

    prob, unc, mask = run_inference(img_rgb)
    overlay         = make_overlay(img_bgr, mask, unc)
    unc_color       = cv2.applyColorMap(
                          (unc * 255).astype(np.uint8),
                          cv2.COLORMAP_JET)
    unc_color       = cv2.cvtColor(unc_color, cv2.COLOR_BGR2RGB)

    # row 0: original
    ax0 = fig.add_subplot(gs[0, col])
    ax0.imshow(img_rgb)
    ax0.axis('off')
    if col == 0:
        ax0.set_ylabel('Input', color='white',
                       fontsize=11, fontweight='bold')
    ax0.set_title(label, color='white', fontsize=9)

    # row 1: overlay
    ax1 = fig.add_subplot(gs[1, col])
    ax1.imshow(overlay)
    ax1.axis('off')
    if col == 0:
        ax1.set_ylabel('Drivable\n(green=safe\nred=uncertain)',
                       color='white', fontsize=9, fontweight='bold')

    # row 2: uncertainty heatmap
    ax2 = fig.add_subplot(gs[2, col])
    ax2.imshow(unc_color)
    ax2.axis('off')
    if col == 0:
        ax2.set_ylabel('Uncertainty\nHeatmap',
                       color='white', fontsize=11, fontweight='bold')

# legend
fig.text(0.5, 0.02,
         '🟢 Drivable  |  🔴 High Uncertainty (boundary/edge cases)  '
         '|  🟦 Uncertainty Heatmap (blue=safe, red=uncertain)',
         ha='center', color='white', fontsize=11)

plt.suptitle('LiteSegEdge — Edge Case Uncertainty Detection\n'
             'Model automatically flags road boundaries, '
             'transitions & ambiguous zones',
             color='white', fontsize=14, fontweight='bold')

plt.savefig(os.path.join(OUT_DIR, 'edge_case_demo.png'),
            dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("Saved to demo/edge_case_demo.png")