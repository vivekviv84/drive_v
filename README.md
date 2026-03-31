\# 🚗 LiteSegEdge — Autonomous Driving Safety Dashboard



A real-time autonomous driving perception system that performs \*\*drivable space segmentation\*\* with \*\*uncertainty estimation\*\* and generates a \*\*risk-aware safety dashboard\*\*.



\---



\## 📌 Overview



This project builds a lightweight deep learning model (\*\*LiteSegEdge\*\*) to identify drivable areas from road images and analyze scene safety using:



\- Semantic segmentation (road vs non-road)

\- Uncertainty estimation (model confidence)

\- Multi-factor risk scoring

\- Real-time visual dashboard



\---



\## 🚀 Key Features



\### ✅ 1. Drivable Area Segmentation

\- Predicts road regions from camera input

\- Lightweight model (\~20K parameters)

\- Real-time inference (200+ FPS on GPU)



\### ✅ 2. Uncertainty Estimation

\- Predicts per-pixel confidence (aleatoric uncertainty)

\- Highlights unreliable regions in the scene



\### ✅ 3. Confidence Tiering

\- 🟢 High confidence (safe)

\- 🟡 Medium confidence (caution)

\- 🔴 Low confidence (uncertain)



\### ✅ 4. Blind Spot Analysis

\- Divides scene into LEFT / CENTER / RIGHT zones

\- Detects unsafe or low-visibility areas



\### ✅ 5. Risk Scoring System

Combines multiple factors:

\- Drivable area coverage

\- Model uncertainty

\- Edge instability

\- Lighting conditions (night detection)



Outputs:

\- 🟢 SAFE

\- 🟡 CAUTION

\- 🔴 HAZARD



\---



\## 🧠 Model Architecture

Input Image → Encoder → Decoder →

├── Segmentation Head (road mask)

└── Uncertainty Head (confidence map)



\- Custom lightweight CNN

\- Skip connections for spatial detail

\- Dual-head output (segmentation + uncertainty)



\---



\## 📊 Dataset



\- \*\*nuScenes (v1.0-mini)\*\*

\- Map-based ground truth generation using drivable area projection

\- Custom mask builder for training labels



\---



\## ⚙️ Tech Stack



\- Python

\- PyTorch

\- OpenCV

\- Albumentations

\- NumPy / Matplotlib



\---



\## 🏗️ Project Structure

LiteSegEdge/

│── models/

│ ├── encoder.py

│ ├── decoder.py

│ └── liteseg\_edge.py

│

│── data/

│── demo/

│ ├── av\_dashboard.py

│ └── demo\_video.py

│

│── mask\_builder\_v4.py

│── train.py

│── metrics.py

│── losses.py

│── README.md



\---



\## 🏃‍♂️ How to Run



\### 1. Install dependencies

```bash

pip install torch torchvision opencv-python albumentations matplotlib

2\. Generate Masks

python mask\_builder\_v4.py

3\. Train Model

python train.py

4\. Run Dashboard

python demo/av\_dashboard.py

5\. Generate Video Demo 🎥

python demo/demo\_video.py

🎥 Demo Output

Real-time drivable area visualization

Risk-aware scene classification

Dynamic safety feedback

📈 Results

mIoU: \~73%

Model size: \~20K parameters

Inference speed: 200+ FPS (GPU)

💡 Key Learnings

Handling noisy segmentation outputs in real-world scenes

Designing uncertainty-aware systems

Combining perception + decision-making in CV pipelines

Importance of data quality over model complexity

🔮 Future Improvements

Lane detection integration

Trajectory prediction

Object detection (vehicles, pedestrians)

Temporal smoothing (video consistency)



👨‍💻 Author



Vivek Dubey





