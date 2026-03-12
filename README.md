# Vision Transformer from Scratch for Urban Change Detection

## 📌 Project Overview

This project focuses on implementing a **Vision Transformer (ViT) from scratch** for the task of **urban growth change detection** using bi-temporal satellite imagery.

Unlike traditional CNN-based change detection models, this project emphasizes:

- Complete mathematical derivation of Vision Transformer
- Implementation of patch embedding and multi-head self-attention from first principles
- Application of transformer-based feature learning to remote sensing imagery
- Performance evaluation for urban change detection

The primary goal is to understand and build the transformer architecture mathematically and programmatically without using pretrained ViT models.

---

## 🌍 Problem Statement

Given two satellite images of the same geographic region captured at different times:

- $\( X\_{T1} \)$
- $\( X\_{T2} \)$

The objective is to predict a binary change map:

$$
\[
Y \in \{0,1\}^{H \times W}
\]
$$

Where:

- 0 → No change
- 1 → Urban growth / structural change

This is formulated as a supervised semantic segmentation problem.

---

## 🧠 Mathematical Formulation

### 1️⃣ Patch Embedding

Given an image:

$$
\[
X \in \mathbb{R}^{H \times W \times C}
\]
$$

Divide the image into patches of size \( P \times P \).

Number of patches:

$$
\[
N = \frac{HW}{P^2}
\]
$$

Each flattened patch is linearly projected into embedding space:

$$
\[
z_0^i = x_p^i E + p_i
\]
$$

Where:

$- \( E \in \mathbb{R}^{(P^2C) \times D} \) is the learnable projection matrix$
$- \( p_i \) is positional encoding$
$- \( D \) is embedding dimension$

---

### 2️⃣ Multi-Head Self-Attention (MHSA)

For token matrix \( Z \):

$$
\[
Q = ZW_Q, \quad K = ZW_K, \quad V = ZW_V
\]
$$

Self-attention is computed as:

$$
\[
Attention(Q,K,V) =
\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]
$$

Multiple heads allow the model to capture different spatial relationships.

---

### 3️⃣ Transformer Encoder Block

Each encoder block consists of:

- Layer Normalization
- Multi-Head Self-Attention
- Residual Connection
- Feed Forward Network (MLP)

$\[Z' = Z + \text{MHSA}(\text{LN}(Z))\]$

$\[ Z\_{out} = Z' + \text{MLP}(\text{LN}(Z'))\]$

---

## 🏗️ Architecture Overview

- Siamese Input (T1 and T2 images)
- Patch Embedding Layer
- Transformer Encoder Stack
- Feature Fusion Layer
- Segmentation Decoder
- Binary Change Map Output

The architecture is designed specifically for bi-temporal remote sensing imagery.

---

## 📊 Dataset

### About:

This dataset contains registered pairs of **13-band multispectral satellite images obtained by the Sentinel-2 satellites of the Copernicus program**. Pixel-level urban change groundtruth is provided. In case of discrepancies in image size, the older images with resolution of 10m per pixel is used. Images vary in spatial resolution between 10m, 20m and 60m. For more information, please refer to Sentinel-2 documentation.

For each location, folders imgs_1_rect and imgs_2_rect contain the same images as imgs_1 and imgs_2 resampled at 10m resolution and cropped accordingly for ease of use.

**The proposed split into train and test images is contained in the train.txt and test.txt files**.

## 📏 Evaluation Metrics

To evaluate change detection performance:

- Intersection over Union (IoU)
- F1 Score
- Precision
- Recall
- Dice Coefficient

---

## 🛠️ Tech Stack

- Python
- PyTorch
- NumPy
- OpenCV
- Matplotlib

---

## 🎯 Project Goals

- Understand Vision Transformer mathematically
- Implement transformer architecture from scratch
- Apply transformer-based learning to urban change detection
- Compare performance with baseline CNN-based approaches

---

## 📌 Course Information

Submitted as part of Machine Learning course project.
