# Deep-Learning-Model-Training-Project-Wildfire-Detection-from-Satellite-Images-
# 🔥 Wildfire Detection & Classification from Satellite Imagery Using Deep Learning

This repository contains the complete implementation, experiments, and evaluation results for the **CENG 476 Introduction to Deep Learning** term project. The objective is to design, train, and rigorously evaluate deep learning models to classify satellite and aerial images for early wildfire detection and post-disaster burn scar identification.

---

## 📊 Dataset Distribution
The project utilizes the publicly available **Kaggle Wildfire Prediction Dataset**. The images are preprocessed, resized to $224 \times 224$ pixels, normalized using ImageNet channel statistics, and split as follows:

| Dataset Split | No Wildfire (Class 0) | Wildfire (Class 1) | Total Images | Percentage |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 14,500 | 15,750 | 30,250 | ~70% |
| **Validation** | 2,820 | 3,480 | 6,300 | ~15% |
| **Test** | 2,820 | 3,480 | 6,300 | ~15% |

---

## 🧠 Model Architectures & Experimental Pipeline
Three distinct architectural approaches were developed and evaluated:
1. **Custom CNN (Baseline from Scratch):** A lightweight convolutional network featuring 3 sequential convolutional blocks with Batch Normalization, ReLU activations, Maxpooling, and an Adaptive Average Pooling layer.
2. **ResNet-18 (Transfer Learning Baseline):** A standard 18-layer residual network leveraging ImageNet pre-trained weights to prevent vanishing gradients.
3. **ConvNeXt-Tiny (Proposed SOTA Model):** A modernized macro-design architecture combining transformer-inspired design principles with standard ConvNet efficiency.

### Regularization & Optimization Strategies:
* **Optimizer:** `AdamW` (Adam with decoupled Weight Decay = $1\hat{-2}$).
* **Learning Rate Scheduler:** `CosineAnnealingLR` to ensure smooth convergence.
* **Regularization:** Batch Normalization, Dropout ($p=0.3 - 0.4$), and extensive **Data Augmentation** (Random flips, rotations, and color jitter).
* **Training Safety:** Automatic Mixed Precision (AMP) via PyTorch `autocast` / `GradScaler` and **Early Stopping** (patience = 4-5 epochs).

---

## 📈 Experimental Results

Independent test set evaluation metrics across all three trained models:

| Model Architecture | Test Accuracy (%) | Macro Precision | Macro Recall | Macro F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN (Scratch)** | 98.05% | 0.9807 | 0.9798 | 0.9802 | 0.9982 |
| **ResNet-18 (Pretrained)** | 99.56% | 0.9953 | 0.9957 | 0.9955 | 0.9998 |
| **ConvNeXt-Tiny (Proposed)** | **99.67%** | **0.9966** | **0.9966** | **0.9963** | **0.9999** |


---

## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/wildfire-prediction-deep-learning.git](https://github.com/YOUR_USERNAME/wildfire-prediction-deep-learning.git)
   cd wildfire-prediction-deep-learning
