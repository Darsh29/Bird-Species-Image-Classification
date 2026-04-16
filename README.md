<div align="center">

# 🦅 Bird Species Image Classification

### Deep learning system for classifying 100 bird species from images

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-a78bfa?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-22c55e?style=for-the-badge)

Benchmarks **4 CNN architectures** on 89,885 images across 100 bird species.
Deep CNN achieves **90.5% accuracy** and **90.4% F1 score**.

</div>

---

## 📊 Results at a Glance

| Model | Architecture | Accuracy | F1 Score | Notes |
|-------|-------------|----------|----------|-------|
| 🥇 Deep CNN | VGG-16 inspired | **90.5%** | **90.4%** | Best overall |
| 🥈 InceptionV3 | Transfer learning | 87.5% | 85.5% | Faster to train |
| 🥉 CNN Base | Shallow custom | ~82% | ~81% | Baseline |
| CNN 512 | Shallow, 150px | ~79% | ~78% | Lower resolution |

> Transfer learning with InceptionV3 trains significantly faster while staying within 3% of the Deep CNN's accuracy.

---

## 🧠 What This Does

This project builds and compares multiple CNN architectures for fine-grained bird species classification. The dataset contains 89,885 labeled images across 100 species sourced under Creative Commons license. All models use the same data augmentation pipeline for fair comparison.

The Deep CNN, inspired by VGG-16's paired convolution blocks, outperforms transfer learning on this dataset likely because the domain-specific features in bird imagery benefit from training from scratch at this dataset scale. InceptionV3 remains competitive with a fraction of the training time.

---

## 🏗️ Model Architectures

### 🔬 Deep CNN (Best Model)
VGG-style paired convolution blocks with progressive filter scaling:

```
Input (224x224x3)
    │
    ▼
Conv2D(32) → Conv2D(32) → MaxPool → BatchNorm
    │
    ▼
Conv2D(64) → Conv2D(64) → MaxPool → BatchNorm
    │
    ▼
Conv2D(128) → Conv2D(128) → MaxPool → BatchNorm
    │
    ▼
Conv2D(256) → Conv2D(256) → MaxPool → BatchNorm
    │
    ▼
Dropout(0.25) → Flatten → Dense(525, softmax)
```

### 🔬 InceptionV3 (Transfer Learning)
Frozen ImageNet weights + custom classification head:

```
InceptionV3 (frozen, no top)
    │
    ▼
GlobalAveragePooling2D
    │
    ▼
Dense(1024, relu) → BatchNorm
    │
    ▼
Dense(525, softmax)
```

### 🔬 CNN Base
Shallow 5-block CNN with progressively wider filters (16→32→64→128→256) at 224x224.

### 🔬 CNN 512
Smaller resolution variant (150x150) with shear augmentation added, faster iteration.

---

## 🗂️ Project Structure

```
Bird-Species-Image-Classification/
├── 📓 cnn_base.ipynb              # Baseline shallow CNN
├── 📓 cnn_deep.ipynb              # Deep VGG-style CNN (best model)
├── 📓 cnn_512_train.ipynb         # Lower-resolution CNN variant
├── 📓 inception_v3.ipynb          # InceptionV3 transfer learning
├── 🐍 data_generation.py          # Dataset loaders and train/val/test splits
└── 📄 requirements.txt
```

> Note: The `data/` directory is not included in the repo due to size (~5GB). See dataset setup below.

---

## 🚀 Getting Started

**Prerequisites:** Python 3.10+, GPU recommended (CPU workable but slow).

```bash
# 1. Clone the repo
git clone https://github.com/Darsh29/Bird-Species-Image-Classification.git
cd Bird-Species-Image-Classification

# 2. Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

Download the dataset from Kaggle: [100 Bird Species](https://www.kaggle.com/datasets/gpiosenka/100-bird-species)

Place it in the following structure:

```
data/
├── train/
│   ├── ALBATROSS/
│   ├── BALD EAGLE/
│   └── ... (100 species folders)
├── test/
│   └── ... (same structure)
└── valid/
    └── ... (same structure)
```

### Run a Model

```bash
# Open any notebook and run all cells
jupyter notebook cnn_deep.ipynb          # Best model
jupyter notebook inception_v3.ipynb      # Transfer learning
jupyter notebook cnn_base.ipynb          # Baseline
jupyter notebook cnn_512_train.ipynb     # Low-res variant
```

---

## 🔄 Data Augmentation Pipeline

All models use the same augmentation strategy during training:

| Technique | Value |
|-----------|-------|
| Rotation | ±30° |
| Brightness | [0.7, 1.3] |
| Zoom | 0.3 |
| Width shift | 0.2 |
| Height shift | 0.2 |
| Horizontal flip | Yes |
| Rescale | 1/255 |

Two data split strategies are available in `data_generation.py`:

- `prepare_data()` — uses the dataset's original train/val/test folders, then re-splits train into 80/20
- `prepare_from_combined()` — merges all data and re-splits 80/15/5 (used by Deep CNN for maximum training data)

---

## 💡 Key Findings

- **Deep CNN beats transfer learning** on this dataset — domain-specific features in bird imagery benefit from training from scratch at 89K images
- **BGE vs FinBERT analog**: InceptionV3's ImageNet weights don't transfer as cleanly to fine-grained bird features as a purpose-built deep CNN
- **Resolution matters less than depth**: CNN 512 at 150px underperforms CNN Base at 224px despite similar architecture
- **Combining train/val/test for re-splitting** gives the Deep CNN ~8% more training data, contributing to its accuracy advantage

---

## 🧰 Tech Stack

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![InceptionV3](https://img.shields.io/badge/InceptionV3-Transfer_Learning-f59e0b?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
