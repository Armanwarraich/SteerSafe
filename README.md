<div align="center">

# 🚗 SteerSafe
### AI-Powered Real-Time Driver Drowsiness Detection

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/Armanwarraich/SteerSafe)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Try it live → [huggingface.co/spaces/Armanwarraich/SteerSafe](https://huggingface.co/spaces/Armanwarraich/SteerSafe)**

![SteerSafe Demo](assets/demo.gif)

</div>

---

## 🚨 The Problem

> **1.35 million people die every year in road accidents globally (WHO, 2023)**
> In India, driver fatigue contributes to **40%+ of highway accidents.**

Drivers cannot reliably detect their own fatigue. By the time a driver notices they are sleepy, their reaction time has already degraded by **50%**. Existing solutions are either reactive (rumble strips), passive (rest reminders), or expensive ($500–2000 in premium vehicles only).

**SteerSafe solves this** — a real-time, camera-based, AI-powered system that monitors the driver continuously and fires an alert within **0.67 seconds** of confirmed drowsiness. Runs on any laptop camera. Zero hardware cost.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **ResNet-152 Model** | Fine-tuned on 84,898 infrared eye images |
| 👁 **Real-time Detection** | Monitors eyes 30 times per second locally |
| ⚡ **Sub-700ms Alert** | Fires within 0.67s of confirmed drowsiness |
| 😮 **Yawn Detection** | Early fatigue signal via MAR metric |
| 🔊 **Audio + Visual Alert** | Pygame alarm + red screen flash |
| 📊 **Live Dashboard** | Streamlit analytics with EAR chart + event log |
| 🌐 **Live Web Deployment** | Gradio app on HuggingFace Spaces |
| 📈 **Session Logging** | Timestamped CSV log of all drowsy events |

---

## 🎯 Model Performance

<div align="center">

| Metric | Score |
|---|---|
| **Test Accuracy** | **99.10%** |
| **F1 Score** | **99.09%** |
| **Precision (Sleepy)** | 99.48% |
| **Recall (Sleepy)** | 98.70% |
| **False Alarm Rate** | 0.5% (43 / 8,591) |
| **Miss Rate** | 1.3% (109 / 8,390) |
| **Inference Speed** | < 50ms per frame (CPU) |
| **Alert Latency** | < 700ms |
| **Training Dataset** | 84,898 images |
| **Test Dataset** | 16,981 images |

</div>

---

## 🏗 Architecture

```
┌─────────────────────────────────────────┐
│         TRAINING (Kaggle T4 GPU)        │
│                                         │
│  MRL Eye Dataset (84,898 images)        │
│           ↓                             │
│  ResNet-152 (ImageNet pretrained)       │
│           ↓                             │
│  Phase 1: Train head (5 epochs)         │
│  Phase 2: Fine-tune layer3+4 (15 epochs)│
│           ↓                             │
│  steersafe.pth → steersafe.onnx         │
└──────────────┬──────────────────────────┘
               │ download model
               ▼
┌─────────────────────────────────────────┐
│         LOCAL (Your Machine)            │
│                                         │
│  Webcam → OpenCV → EAR/MAR Detection    │
│           ↓                             │
│  ResNet-152 Inference (ONNX/PyTorch)    │
│           ↓                             │
│  Temporal logic: 20 consecutive frames? │
│           ↓                             │
│  Alert + Streamlit Dashboard            │
└──────────────┬──────────────────────────┘
               │ optional
               ▼
┌─────────────────────────────────────────┐
│      DEPLOYMENT (HuggingFace Free)      │
│                                         │
│  Gradio + OpenCV + ONNX Runtime         │
│  Live webcam via browser                │
│  Public URL — zero cost                 │
└─────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Category | Technology | Why Chosen |
|---|---|---|
| **Deep Learning** | PyTorch + ResNet-152 | Dynamic graphs, easy layer freezing for transfer learning |
| **Model Export** | ONNX Runtime | 25% faster inference, no PyTorch dependency on server |
| **Computer Vision** | OpenCV | Real-time webcam capture, frame processing, eye detection |
| **Face Landmarks** | dlib (local) | 68-point model maps directly to EAR formula indices |
| **Web UI** | Gradio | Native HuggingFace support, built-in webcam streaming |
| **Dashboard** | Streamlit | Python-only, built-in charts, zero frontend knowledge needed |
| **Alert Audio** | Pygame | Non-blocking audio playback in detection loop |
| **Training Cloud** | Kaggle (T4 GPU) | 30hrs/week free, MRL dataset pre-hosted, 15.6GB VRAM |

---

## 📁 Repository Structure

```
SteerSafe/
│
├── src/
│   ├── detection.py          # Main local webcam detection loop
│   └── ear_utils.py          # EAR + MAR geometric calculations
│
├── app/
│   ├── streamlit_app.py      # Local analytics dashboard
│   └── gradio_app.py         # HuggingFace deployment app
│
├── assets/
│   └── alarm.wav             # Alert sound (auto-generated)
│
├── requirements.txt
└── README.md

# Not in this repo (too large for GitHub):
# models/steersafe.pth        → trained weights (~250MB)
# models/steersafe.onnx       → ONNX export (~250MB)
# models/shape_predictor_68_face_landmarks.dat → dlib model (~99MB)
# hf_deploy/                  → separate HuggingFace git repo
```

---

## 🚀 Local Setup

### Prerequisites
- Python 3.11+
- Webcam
- Windows / Linux / Mac

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Armanwarraich/SteerSafe.git
cd SteerSafe

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Download Required Model Files

| File | Size | Download |
|---|---|---|
| `steersafe.pth` | ~250MB | [Kaggle Output](https://www.kaggle.com) |
| `steersafe.onnx` | ~250MB | [Kaggle Output](https://www.kaggle.com) |
| `shape_predictor_68_face_landmarks.dat` | ~99MB | [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) |

Place all files in `models/` folder.

### Generate Alarm Sound

```bash
python -c "
import numpy as np, wave
t = np.linspace(0, 1.5, 66150)
d = (np.sin(2*np.pi*880*t)*32767).astype('int16')
with wave.open('assets/alarm.wav','w') as f:
    f.setnchannels(1); f.setsampwidth(2)
    f.setframerate(44100); f.writeframes(d.tobytes())
print('alarm.wav created!')
"
```

### Run

```bash
# Live webcam detection (smooth, real-time, with audio)
python src/detection.py

# Analytics dashboard (open in second terminal)
streamlit run app/streamlit_app.py
# Opens at http://localhost:8501
```

---

## 🌐 HuggingFace Deployment

The live demo is hosted at:
**[huggingface.co/spaces/Armanwarraich/SteerSafe](https://huggingface.co/spaces/Armanwarraich/SteerSafe)**

### What's different on HuggingFace vs local

| Feature | Local | HuggingFace |
|---|---|---|
| Face detection | dlib 68-point landmarks | OpenCV Haar Cascade |
| Model format | PyTorch `.pth` | ONNX `.onnx` |
| Alert sound | Pygame audio | Visual red flash |
| Latency | Real-time (~30fps) | ~2-3s (free CPU) |
| Access | Your machine only | Public URL worldwide |

### Files on HuggingFace (separate repo, not here)
```
huggingface.co/spaces/Armanwarraich/SteerSafe
├── app.py
├── steersafe.onnx          (via Git LFS)
├── steersafe.onnx.data     (via Git LFS)
└── requirements.txt
```

> **Note:** Model files are stored on HuggingFace via Git LFS.
> They are not in this GitHub repo due to GitHub's 100MB file limit.

---

## 🧠 How It Works

### 1. Eye Aspect Ratio (EAR)
The core geometric metric for drowsiness detection:

```
         p2    p3
    p1              p4
         p6    p5

EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)

Open eye  → EAR ≈ 0.25-0.30
Closing   → EAR drops toward 0
Alert     → EAR < 0.25 for 20+ consecutive frames
```

### 2. Two-Phase Transfer Learning

```
Phase 1 (5 epochs):
  ResNet-152 base → FROZEN
  Custom head     → TRAINING
  Why: Prevent random gradients from corrupting ImageNet weights
  Result: Val accuracy 95.82%

Phase 2 (15 epochs):
  layer1, layer2  → FROZEN
  layer3, layer4  → FINE-TUNING (lr=1e-4)
  Custom head     → FINE-TUNING
  Why: Gently adapt deep features to eye-specific patterns
  Result: Val accuracy 99.06%
```

### 3. Dual-Signal Alert System
```
Signal 1: ResNet-152 → "Sleepy" (image classification)
Signal 2: EAR < 0.25 (geometric eye closure)

Either signal → increment drowsy counter
Counter ≥ 20 frames → ALERT FIRES

Why dual? CNN misses are caught by EAR geometry.
Both must fail simultaneously for a true miss.
```

---

## 📊 Training Details

| Parameter | Value |
|---|---|
| Dataset | MRL Eye Dataset |
| Training images | 50,937 |
| Validation images | 16,980 |
| Test images | 16,981 |
| Base model | ResNet-152 (ImageNet) |
| Phase 1 LR | 1e-3 |
| Phase 2 LR | 1e-4 |
| Batch size | 64 |
| Total epochs | 20 (5 + 15) |
| Training hardware | Kaggle Tesla T4 (15.6GB) |
| Training time | ~5 hours |
| Optimizer | Adam + ReduceLROnPlateau |

---

## 🔮 Future Improvements

- [ ] **CBAM Attention** — Add Spatial Attention on top of Channel Attention for better eye-region focus
- [ ] **Night mode** — CLAHE histogram equalisation for dark driving conditions
- [ ] **Glasses support** — MediaPipe fallback for glasses wearers
- [ ] **CEW Dataset** — Add Closed Eyes in the Wild dataset to reduce false negatives
- [ ] **Grad-CAM** — Visual explainability showing which pixels the model focuses on
- [ ] **More epochs** — Val accuracy still climbing at epoch 15, 25 epochs may hit 99.5%+
- [ ] **Head pose estimation** — Detect nodding off even before eye closure
- [ ] **Vehicle integration** — CAN bus integration for automated safety response
- [ ] **Mobile app** — TensorFlow Lite export for Android deployment

---

## 🤝 Contributing

Contributions are welcome!

```bash
# 1. Fork the repository
# 2. Create your feature branch
git checkout -b feature/your-feature

# 3. Commit your changes
git commit -m "Add your feature"

# 4. Push to the branch
git push origin feature/your-feature

# 5. Open a Pull Request
```

---

## 📋 Problems Faced & Solutions

| Problem | Solution | Learning |
|---|---|---|
| Colab compute limit exhausted | Switched to Kaggle (30hrs/week free) | Always check compute limits upfront |
| 85K image upload taking 4+ hours | Used Kaggle pre-hosted dataset | Use platforms where data already exists |
| Wrong dataset paths on Kaggle | Used `os.walk()` to inspect structure | Always verify paths before training |
| PyTorch 2.10 scheduler API change | Removed deprecated `verbose=True` | Check API compatibility for new environments |
| ONNX export dependency missing | `pip install onnxscript` + `dynamo=False` | Pin dependencies in requirements |
| MediaPipe no OpenGL on HuggingFace | Switched to OpenCV Haar Cascade | Match library to deployment environment |
| dlib compile failure on HuggingFace | Replaced with OpenCV (no compilation) | Prefer pre-built wheels for cloud deployment |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Arman Warraich**
B.Tech CSE Final Year | AI/ML Domain

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Armanwarraich-FFD21E?style=flat-square)](https://huggingface.co/Armanwarraich)
[![GitHub](https://img.shields.io/badge/GitHub-Armanwarraich-181717?style=flat-square&logo=github)](https://github.com/Armanwarraich)

---

<div align="center">

**⭐ Star this repo if you found it useful!**

*SteerSafe — Because every driver deserves to arrive safely.*

</div>
