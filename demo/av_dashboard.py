import torch
import cv2
import numpy as np
import os, sys
import matplotlib.pyplot as plt
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

# ── CONFIDENCE ─────────────────────────────
def confidence_tier_overlay(img_rgb, prob, unc, mask):
    overlay = img_rgb.copy()

    drivable_mask = mask > 0
    if drivable_mask.any():
        unc_vals = unc[drivable_mask]
        t_high = np.percentile(unc_vals, 70)
        t_low  = np.percentile(unc_vals, 95)
    else:
        t_high, t_low = 0.4, 0.7

    high = ((mask>0)&(unc<=t_high)).astype(np.uint8)
    med  = ((mask>0)&(unc>t_high)&(unc<=t_low)).astype(np.uint8)
    low  = ((mask>0)&(unc>t_low)).astype(np.uint8)

    for m,c in [(high,(0,220,80)),(med,(255,200,0)),(low,(220,50,50))]:
        color = np.zeros_like(img_rgb)
        color[:,:,0],color[:,:,1],color[:,:,2]=c
        overlay = np.where(np.stack([m]*3,2),
                           cv2.addWeighted(overlay,0.45,color,0.55,0),
                           overlay)

    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay,contours,-1,(0,255,255),2)

    return overlay,high,med,low


# ── BLIND SPOTS ───────────────────────────
def draw_blind_spots(overlay, mask, unc):
    h,w=overlay.shape[:2]
    zones={'LEFT':(0,0,w//4,h),'CENTER':(w//4,0,3*w//4,h),'RIGHT':(3*w//4,0,w,h)}
    warnings=0

    for name,(x1,y1,x2,y2) in zones.items():
        zm=mask[y1:y2,x1:x2]
        zu=unc[y1:y2,x1:x2]

        cov=zm.mean()
        mu=zu[zm>0].mean() if zm.any() else 1

        if cov>0.45 and mu<0.4:
            col,st=(0,220,80),'CLEAR'
        elif cov>0.2:
            col,st=(255,200,0),'CAUTION'
        else:
            col,st=(220,50,50),'WARNING'
            warnings+=1

        cv2.rectangle(overlay,(x1,y1),(x2,y2),col,2)
        cv2.putText(overlay,name,(x1+5,y2-30),0,0.5,col,2)
        cv2.putText(overlay,st,(x1+5,y2-10),0,0.5,col,1)

    return overlay,warnings


# ── RISK ──────────────────────────────────
def compute_risk(prob, unc, low, warnings, img):

    mask=prob>0.68
    dp=mask.mean()*100

    cov_risk=(1-dp/100)**2*40

    p75=np.percentile(unc[mask],75) if mask.any() else 1
    unc_risk=p75*15   # 🔥 reduced impact

    edge=(low.sum()/max(mask.sum(),1))*20
    blind=min(15,warnings*5)

    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    bright=gray.mean()

    night=0
    if bright<120:
        night=(120-bright)/120*40

    score=int(min(100,cov_risk+unc_risk+edge+blind+night))

    # 🔥 SAFE BOOST
    if dp>45 and p75<0.3:
        score -= 10

    if score>=60: return score,'HAZARD',(220,50,50),dp
    elif score>=30: return score,'CAUTION',(255,200,0),dp
    else: return score,'SAFE',(0,220,80),dp


# ── MAIN ──────────────────────────────────
nusc=NuScenes(version=VERSION,dataroot=DATAROOT,verbose=False)
indices=[0,40,80,120,200,300]

fig,axes=plt.subplots(2,3,figsize=(20,12))
axes=axes.flatten()

print("FINAL RUN...")

for i,idx in enumerate(indices):

    s=nusc.sample[idx]
    cam=nusc.get('sample_data',s['data']['CAM_FRONT'])
    img=cv2.imread(os.path.join(DATAROOT,cam['filename']))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(640,360))

    t=transform(image=img)['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        seg,lv=model(t)

    prob=torch.sigmoid(seg).squeeze().cpu().numpy()
    unc =torch.sigmoid(lv).squeeze().cpu().numpy()

    # 🔥 FINAL UNC FIX
    unc=cv2.GaussianBlur(unc,(7,7),0)
    unc=np.clip(unc,0,0.6)

    # 🔥 FINAL MASK
    mask=(prob>0.68).astype(np.uint8)
    h,w=mask.shape
    mask[:int(h*0.45),:]=0
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((7,7),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))

    prob=prob*mask

    overlay,high,med,low=confidence_tier_overlay(img,prob,unc,mask)
    overlay,warnings=draw_blind_spots(overlay,mask,unc)

    score,label,color,dp=compute_risk(prob,unc,low,warnings,img)

    cv2.putText(overlay,f"{label} {score}/100",(10,30),0,0.8,color,2)

    axes[i].imshow(overlay)
    axes[i].axis('off')
    axes[i].set_title(f"Frame {idx} | {label}",color='white')

    print(f"Frame {idx}: {label} {score}")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'av_dashboard.png'))
plt.show()