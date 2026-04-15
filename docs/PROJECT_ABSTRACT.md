# PROJECT ABSTRACT

## Project Title

**Vision Transformer from Scratch for Urban Change Detection**

---

## Project Abstract

This project implements a **Vision Transformer (ViT) from scratch** for the task of **urban growth change detection** using bi-temporal satellite imagery. Unlike traditional CNN-based change detection models, the project emphasizes complete mathematical derivation and implementation of the Vision Transformer architecture from first principles—including patch embedding, multi-head self-attention, and transformer encoder blocks—without using pretrained models.

Given two satellite images of the same geographic region captured at different times (T1 and T2), the objective is to predict a binary change map where 0 indicates no change and 1 indicates urban growth or structural change. This is formulated as a supervised semantic segmentation problem.

The implementation builds a Siamese-style architecture with patch embedding, a transformer encoder stack, feature fusion, and a segmentation decoder to produce binary change maps. The project also includes ablation studies (e.g., pre-norm vs. post-norm, [CLS] token vs. global average pooling) and supports both CIFAR-10/EuroSAT configurations and urban change detection datasets.

---

## Scope of Work

1. **Mathematical Formulation & Derivation**
   - Patch embedding: dividing images into patches and linear projection
   - Multi-head self-attention (MHSA) with Q, K, V projections
   - Transformer encoder blocks (LayerNorm, MHSA, residual connections, MLP)

2. **Model Implementation**
   - Patch embedding layer
   - Multi-head self-attention module
   - Transformer encoder blocks (pre-norm, post-norm, no-residual variants)
   - Full ViT model assembly with classification/segmentation head

3. **Data Pipeline**
   - CIFAR-10 and EuroSAT data loaders
   - Data augmentation (RandomCrop, RandAugment, RandomErasing)
   - Bi-temporal urban change detection dataset support (LEVIR-CD, WHU-CD)

4. **Training & Evaluation**
   - Training pipeline with warmup and cosine decay
   - Evaluation metrics: IoU, F1, Precision, Recall, Dice Coefficient
   - Ablation studies (scaling, [CLS] vs. GAP, block variants)

5. **Visualization & Analysis**
   - Attention map visualization
   - t-SNE embeddings for learned representations

---

## Split of Work amongst Team Members

*[To be filled in by the team]*

| Team Member | Responsibility |
|-------------|----------------|
| Member 1    | *e.g., Model architecture, patch embedding, attention* |
| Member 2    | *e.g., Data pipeline, augmentation, dataset loading* |
| Member 3    | *e.g., Training loop, evaluation metrics, experiments* |
| Member 4    | *e.g., Visualization, ablation studies, documentation* |

---

## Tech Stack

- **Language:** Python
- **Framework:** PyTorch
- **Libraries:** NumPy, OpenCV, Matplotlib

---

## Datasets

- LEVIR-CD (Building Change Detection)
- WHU-CD
- CIFAR-10 (for initial validation)
- EuroSAT (satellite imagery)

---

## Course Information

Submitted as part of the Machine Learning course project.
