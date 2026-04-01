# 🚗 LiteSegEdge — Autonomous Driving Perception & Safety Dashboard
*A lightweight real-time perception system that segments drivable space, estimates model uncertainty, and generates a risk-aware safety dashboard from standard camera input.*

---

## 📌 What We Built

LiteSegEdge started from a straightforward question: **how do you make a self-driving system honest about what it doesn't know?**

Most segmentation models tell you where the road is. This one also tells you *how confident it is* — and flags the parts of the scene it isn't sure about. The result is a system that not only identifies drivable areas but generates a live safety assessment combining perception quality, lighting conditions, and scene stability.

Built and trained on the **nuScenes mini dataset**, the model is deliberately small (~20K parameters) to stay fast enough for real-time use without sacrificing the uncertainty estimation that makes the dashboard meaningful.

---

## 🧠 Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        LiteSegEdge Platform                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   nuScenes   │───▶│ Custom Light │───▶│  Dual-Head   │       │
│  │   Dataset    │    │ Weight CNN   │    │   Outputs    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                  │               │
│                                                  ▼               │
│                                 ┌──────────────────────────────┐│
│                                 │ Segmentation + Uncertainty   ││
│                                 └──────────────────────────────┘│
│                                                  │               │
│                                                  ▼               │
│  ┌─────────────────────────────────────────────────────┐        │
│  │            Risk Assessment & Zone Analysis           │        │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │        │
│  │  │Drivable │ │  Conf.  │ │  Blind  │ │Lighting │  │        │
│  │  │Coverage │ │ Tiering │ │  Spot   │ │  Check  │  │        │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘  │        │
│  │       └───────────┴───────────┴───────────┘        │        │
│  │                   Multi-Factor Risk Scoring         │        │
│  └─────────────────────────────────────────────────────┘        │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────┐        │
│  │         AV Safety Dashboard (OpenCV UI)              │        │
│  │  ┌─────────┐ ┌───────────┐ ┌─────────┐ ┌─────────┐ │        │
│  │  │  Road   │ │Uncertainty│ │  Zone   │ │  Risk   │ │        │
│  │  │ Overlay │ │    Map    │ │ Overlay │ │ Status  │ │        │
│  │  └─────────┘ └───────────┘ └─────────┘ └─────────┘ │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Model Design

```text
Input Image
    └── Encoder (lightweight CNN with skip connections)
         └── Decoder
              ├── Segmentation Head  →  road mask
              └── Uncertainty Head   →  confidence map
```

The dual-head design shares a single encoder, so both outputs are computed in one forward pass with no meaningful latency penalty. Skip connections preserve spatial detail that would otherwise be lost during downsampling.

---

## 🚀 Key Features

### 1. High-Speed Drivable Area Segmentation
Predicts road vs. non-road regions from a single camera frame. Runs at 200+ FPS on GPU — fast enough to keep up with real driving scenarios.

### 2. Live Uncertainty Mapping
Alongside the segmentation mask, the model outputs a confidence map highlighting which parts of the prediction are unreliable. A model that's wrong but confident is dangerous; one that's wrong and *knows it* gives the system a chance to respond.

### 3. Confidence Tiering System

| Status | Confidence | Description |
| :--- | :--- | :--- |
| 🟢 **High** | > 80% | Safe to navigate; highly reliable predictions |
| 🟡 **Medium** | 50% – 80% | Exercise caution; manual takeover may be required |
| 🔴 **Low** | < 50% | Highly uncertain; engage failsafe protocols |

### 4. Blind Spot Zone Analysis
The scene is divided into **left, center, and right zones**. Each zone is independently checked for visibility and safety, surfacing issues that a single scene-wide score would miss.

### 5. Composite Risk Scoring
A final risk score is computed from four factors: drivable area coverage, model uncertainty, edge instability, and a night-detection heuristic for lighting. The output is one of three states — **`SAFE`**, **`CAUTION`**, or **`HAZARD`** — displayed on the dashboard in real time.

### 6. Lane Boundary Detection *(newly integrated)*
Detects lane markings and boundaries to complement drivable area segmentation, giving the system a clearer picture of legal driving corridors within the road surface.

### 7. Trajectory Prediction *(newly integrated)*
Forecasts the short-term movement of detected objects, enabling the risk scoring system to account not just for where things are, but **where they're headed**.

### 8. Temporal Smoothing *(newly integrated)*
Applies frame-to-frame consistency filtering across the video stream, reducing flicker in segmentation outputs and producing more stable, reliable dashboard readings over time.

---

## 🏃‍♂️ Quick Start

### Prerequisites
- Ubuntu 20.04+ or macOS 12+
- NVIDIA GPU recommended (for 200+ FPS inference)
- Python 3.8+

### Setup & Execution
```bash
# Clone and enter the repository
git clone https://github.com/your-username/LiteSegEdge.git
cd LiteSegEdge

# Install dependencies
pip install torch torchvision opencv-python albumentations matplotlib

# 1. Generate map-based ground truth masks
python mask_builder_v4.py

# 2. Train the lightweight model
python train.py

# 3. Run the live dashboard
python demo/av_dashboard.py

# 4. Generate the video demo
python demo/demo_video.py
```

### 🎯 Execution Entry Points
LiteSegEdge runs purely as optimized local Python scripts using OpenCV and Matplotlib for real-time visual popups. No web servers required.

| Command | Action | Output Type |
| :--- | :--- | :--- |
| `python demo/av_dashboard.py` | Full safety dashboard | Live OpenCV Window |
| `python train.py` | Runs curriculum training loop | Terminal stdout + logs |
| `python utils/benchmark.py` | Evaluates system FPS | Matplotlib Popup |
| `python demo/edge_case_demo.py` | Uncertainty edge-case analyzer | Matplotlib Popup |

---

## 📁 Project Structure

```text
LiteSegEdge/
├── models/
│   ├── encoder.py
│   ├── decoder.py
│   └── liteseg_edge.py
├── demo/
│   ├── av_dashboard.py
│   └── demo_video.py
├── losses/
├── utils/
├── mask_builder_v4.py
├── train.py
├── metrics.py
└── README.md
```

---

## 📊 Performance Results

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **mIoU** | ~73% | Highly competitive for a compact model |
| **Model Size** | ~20K params | Extreme edge-device efficiency |
| **Inference Speed** | 283 FPS | Benchmarked on PyTorch CUDA (RTX 3070 Ti) |
| **Exported Speed** | 171 FPS | Benchmarked on ONNX Runtime |

---




## 🔬 ML Pipeline Details

### ⚙️ Model Hyperparameters

| Parameter | Value |
| :--- | :--- |
| **Learning Rate** | `1e-3` (cosine decay to 0) |
| **Batch Size** | `8` |
| **Optimizer** | AdamW |
| **Epochs** | `60` (3-phase curriculum) |
| **Image Size** | `640 × 360` |
| **Total Parameters** | `20,226` |

### 🖼️ Data Augmentations (Albumentations)

| Transform | Probability |
| :--- | :--- |
| `HorizontalFlip` | p=0.5 |
| `RandomBrightnessContrast` | p=0.3 |
| `GaussianBlur` | p=0.2 |
| `ShiftScaleRotate` | p=0.3 |
| `Normalize` | ImageNet mean/std |

### 📐 Multi-Task Loss Strategy

| Loss Component | Weight (λ) | Purpose |
| :--- | :--- | :--- |
| **Focal Loss** | 1.0 | Battles background/road class imbalance |
| **Lovász Binary Loss** | 0.5 | Directly optimizes the IoU metric |
| **Uncertainty Loss** | 0.3 | Calibrates the auxiliary uncertainty head |
| **Boundary-Aware Loss** | 0.5 | Penalizes bleeding errors near road edges |

### 🗺️ Dataset
**nuScenes v1.0-mini** — Ground truth masks are generated using the nuScenes map API, projecting drivable area annotations into the camera frame. A custom mask builder (`mask_builder_v4.py`) handles this projection and produces the binary labels used during training.

---

## 💡 What We Learned

- **Uncertainty estimation changes how you think about model outputs entirely.** A confidence map isn't just a nice visual — it's a second opinion on every prediction.
- **Data quality had a bigger impact on final mIoU than any architectural change we tried.** Getting the mask projection right was most of the battle.
- **Combining perception with a decision layer** is where the project stopped feeling like a segmentation exercise and started feeling like an actual safety system.
- **Handling noisy outputs in real-world scenes** requires more than a good model — it requires thinking carefully about what the model is *allowed* to be uncertain about.

---

## 🔮 Future Work
- **Object Detection** — Identifying and localizing dynamic agents (vehicles, pedestrians, cyclists) to add a critical layer of awareness beyond static road geometry.

---

## 👥 Team
Built  by a 4-person team:

| Name | Role |
| :--- | :--- |
| **Neha Shetty** | Led project planning, coordination, and timeline management across all team members. Owned documentation, presentation flow, and integration oversight |
| **Vivek Dubey** | Designed and built LiteSegEdge (encoder-decoder + uncertainty pipeline), custom loss functions (Focal + Lovász + Boundary + Uncertainty), and nuScenes mask generation. Developed the real-time risk scoring dashboard with multi-factor evaluation, optimized to 200+ FPS. |
| **Prakati N** | Handled dataset preparation, preprocessing, and augmentation; ran training experiments and parameter tuning to evaluate model performance.
 || **Shreya Sanghwa** | Assisted in dashboard visualization design and demo video creation; led report writing and PPT preparation for result presentation. 
---

## 📄 License
This project is open-sourced under the **[MIT License](LICENSE)** — feel free to use, modify, and learn from it.

---

*Built for learning, experimentation, and the belief that safer autonomy starts with knowing what you don't know.*
