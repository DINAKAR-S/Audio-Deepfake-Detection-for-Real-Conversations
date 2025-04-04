# 🎙️ Audio Deepfake Detection using RawNet2

This repository contains an implementation of **RawNet2**, an end-to-end deep learning model for detecting AI-generated (spoofed) human speech. The model is trained and evaluated on the ASVspoof 2019 Logical Access (LA) dataset.

---

## 🔍 Project Overview

- **Goal:** Detect AI-generated (deepfake) speech in real-time or near real-time using efficient deep learning models.
- **Use Case:** Analyze real-world conversations and voice recordings to flag fake (TTS/VC-generated) audio.
- **Model Implemented:** RawNet2 (ICASSP 2021)

---

## 📌 Key Features

- End-to-end training on raw waveform inputs.
- Utilizes Sinc filters and GRU-based structure to extract robust temporal features.
- Compatible with ASVspoof 2019 LA dataset.
- CPU-friendly implementation for low-resource environments.

---

### 📦 Python Environment

**Python Version:** `3.6` or higher (tested with Python 3.10)

### 📁 Project Directory Structure

```
audio-deepfake-rawnet2/
│
├── data/
│   └── LA/
│       ├── ASVspoof2019_LA_train/
│       ├── ASVspoof2019_LA_dev/
│       ├── ASVspoof2021_LA_eval/
│       └── protocols/
├── main.py
├── model.py
├── data_utils.py
├── model_config_RawNet.yaml
├── requirements.txt
├── README.md
└── models/ (generated after training)
```

---

### 🔧 Setup Instructions

1. **Clone the Repository**
```bash
git clone https://github.com/DINAKAR-S/Audio-Deepfake-Detection-for-Real-Conversations.git
cd audio-deepfake-rawnet2
```

2. **Create and Activate Virtual Environment**
```bash
# Create venv
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

3. **Install Required Packages**
```bash
pip install -r requirements.txt
```

---

### 📄 `requirements.txt` (Sample Content)

```
torch==1.12.1
librosa
numpy
PyYAML
tensorboardX
scikit-learn
```

> ✅ **Note**: You can install the latest compatible version of `torch` if `1.12.1` fails on your system:
```bash
pip install torch
```

---

### 📥 Dataset Setup (ASVspoof 2019 LA)

1. Visit [ASVspoof 2019 Dataset](https://datashare.is.ed.ac.uk/handle/10283/3336)
2. Download the following:
   - `ASVspoof2019_LA_train.zip`
   - `ASVspoof2019_LA_dev.zip`
   - `ASVspoof2021_LA_eval.zip`
   - Protocol files from: `ASVspoof2019_LA_cm_protocols` and `ASVspoof2021_LA_cm_protocols`
3. Extract all under:  
   `./data/LA/`

---

### 📌 Reproducibility Notes

- Set seed using `--seed` flag to ensure reproducibility:
```bash
--seed 1234
```
- Ensure all three datasets (`train`, `dev`, `eval`) are in the same root path.
- Paths can be customized in `main.py` or passed via CLI:
```bash
--database_path ./data/LA/ --protocols_path ./data/LA/
```

---

## 🚀 Training & Evaluation

### ▶️ Train the Model
```bash
python main.py --database_path ./data/LA/ --protocols_path ./data/LA/ --num_epochs 5
```

### 📊 Evaluate
```bash
python main.py --eval --model_path models/model_LA_weighted_CCE_5_8_0.0001/epoch_4.pth \
--eval_output results/eval_scores.txt --database_path ./data/LA/ --protocols_path ./data/LA/
```

---

## 📈 Results (Example)

| Metric        | Value     |
|---------------|-----------|
| Train Accuracy | 100%      |
| Validation Accuracy | ~10.2% (early stopping) |
| Epochs        | 5         |
| Device Used   | CPU (no GPU) |

---

## 🧠 Model Summary

RawNet2 is a lightweight end-to-end model that:
- Uses SincConv filters to process raw waveforms.
- Learns discriminative embeddings via GRU + Fully connected layers.
- Performs binary classification (bonafide vs spoofed).

---

## 💡 Future Improvements

- Add data augmentation (e.g., noise injection with MUSAN).
- Try other models (AASIST, ResMax) for comparison.
- Deploy real-time inference pipeline via Streamlit or FastAPI.

---
