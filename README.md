[Video Input] ──> [MediaPipe Landmark Detection (40 Lip Points)]
──> [Geometric Vector Extraction & Normalization]
──> [Production-Grade Spatial-Temporal Factory Model]
──> [14-Class Word Prediction]

---

## 🔬 Methodology & Feature Engineering

### 1. Dataset & Preprocessing
* **Scope:** Focused on the Indonesian language, covering a structured vocabulary of 14 distinct words (representing Subject, Predicate, and Object combinations).
* **Data Scale:** A comprehensive private dataset comprising **5,600 video sequences** captured across 40 distinct subjects (11 males, 29 females).
* **Temporal Alignment:** Video durations are standardized via interpolation/truncation to exactly **44 frames** (approx. 1.5 seconds maximum).

### 2. Geometric Vector Extraction
* Using **MediaPipe Face Mesh**, 40 highly descriptive landmarks surrounding the inner and outer lip contours are extracted dynamically.
* **Vector Generation:** A 1D geometric vector is constructed from each of the 40 lip landmarks to a predefined reference point. Three reference points were extensively evaluated: **Mouth Center, Nose, and Chin**.
* **Normalization Engine:** * *Scale Rigidity:* Vector lengths are normalized against the dynamically calculated inter-ocular distance to handle speaker-to-camera distance variations.
  * *Orientation Rigidity:* Spatial coordinates are stabilized via an orthogonal vector matrix derived from the forehead, nose, and eyes to nullify head-tilt variations.
  * *Feature Selection:* Each landmark tracking point yields 4 key descriptors: $v_x$ (horizontal velocity), $v_y$ (vertical velocity), magnitude, and angle.

---

## ⚙️ Model Architecture

The core framework is built upon a unified **Modular Spatial-Temporal Factory Pattern** designed using the Keras Functional API to map local spatial frame features and global sequence dynamics.

* **Spatial Blocks (1D-CNN):** A 3-layer deep 1D Convolutional network utilizing increasing filter sizes (64 -> 128 -> 256) combined with Batch Normalization, L2 regularization, and progressive Dropout to capture local spatial relationships along the feature vector dimensions.
* **Temporal Blocks (RNN Hybrids):** Captures time-series trends using Gated Recurrent Units (GRU) and Long Short-Term Memory (LSTM) layers, utilizing `Bidirectional` wrapping to analyze both past and future contextual frames simultaneously.

---

## 📊 Experimental Results

Our empirical evaluations demonstrate that representing lip dynamics via **Absolute Per-Frame Positions** provides far more discriminative geometric features than tracking frame-to-frame delta differences.

### Reference Point & Model Comparison

| Reference Point | Optimal Architecture | Testing Accuracy | F1-Score |
| :--- | :--- | :---: | :---: |
| 🎯 **Mouth Center (Optimal)** | **1DCNN - biLSTM (Per-Frame)** | **96.96%** | **96.96%** |
| 👃 Nose | 1DCNN - biLSTM | > 90.00% | > 90.00% |
| 🦩 Chin | 1DCNN - biLSTM | > 90.00% | > 90.00% |

*The **Mouth Center** acts as the perfect structural anchor since it sits directly at the focal axis of lip articulation, producing highly distinct vector distributions for homophenes.*

---

## 🚀 Getting Started

### 1. Environment Setup
Clone the repository and install dependencies using a modern, fast package manager like `uv` (or standard `pip` inside a virtual environment):

```bash
# Initialize environment
uv venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Install locked dependencies
uv pip install -r requirements.txt

### 2. Running the Training Pipeline
Verify the end-to-end data loading, normalization, and training loop using the baseline 1DCNN model and the enclosed toy dataset:

python src/train.py \
  --data_path "sample_data" \
  --model 1DCNN \
  --reference mouth \
  --epochs 5 \
  --batch_size 16

3. Running Model Evaluation (Inference)
Evaluate a trained serialization file (.h5) to output comprehensive confusion matrices, classification reports, and class-wise ROC AUC scores:

python src/predict.py \
  --model_path "models/2026-05-30-1DCNN-mouth.h5" \
  --data_path "sample_data" \
  --reference mouth

📂 Repository Structure

LipMovementRecognition/
│
├── sample_data/           # Truncated sample dataset for pipeline verification
│   └── vector/
│       └── mouth/
│           ├── air/       # Contains minimal sample .npy files
│           ├── aku/
│           └── ...        # Subfolders for all 14 vocabulary classes
│
├── src/                   # Production-grade source modules
│   ├── __init__.py
│   ├── data_loader.py     # Streamlined tf.data pipeline with per-video Standard Scaling
│   ├── models.py          # Unified Functional API Architecture Factory
│   ├── train.py           # Core execution script for model fitting and history plotting
│   └── predict.py         # Advanced evaluation script yielding percentage-scaled metrics
│
├── models/                # Local directory for saving compiled weight checkpoints (.h5)
├── requirements.txt       # Environment dependency constraints (Optimized for Python 3.12)
└── README.md              # Project documentation

📖 Citation / Reference

This repository is part of the research published and presented at the International Conference on Information Technology Systems and Innovation (ICITSI). If you find this codebase or methodology useful in your research, please cite our paper:

Paper Title: Lip Movement Recognition Based on Keypoint Vector using Deep Learning
Publisher: IEEE Xplore
Link: https://ieeexplore.ieee.org/abstract/document/10929448
DOI: 10.1109/ICITSI65188.2024.10929448