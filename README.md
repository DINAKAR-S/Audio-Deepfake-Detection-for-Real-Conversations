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

## 📚 Dataset

We use the [ASVspoof 2019 Logical Access](https://doi.org/10.7488/ds/2555) dataset.

Directory structure:
```
data/
├── LA/
│   ├── ASVspoof2019_LA_train/
│   ├── ASVspoof2019_LA_dev/
│   ├── ASVspoof2021_LA_eval/
│   └── protocols/
```

---

## 🛠️ Installation

```bash
# Clone repo
git clone https://github.com/<your-username>/audio-deepfake-rawnet2.git
cd audio-deepfake-rawnet2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
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
