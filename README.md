# Conditional GANs with WGAN-GP  
*CSL7590: Deep Learning | Spring 2024 | IIT Jodhpur*

![GitHub repo size](https://img.shields.io/github/repo-size/yourusername/Conditional-GAN-WGAN-GP)
![License](https://img.shields.io/github/license/yourusername/Conditional-GAN-WGAN-GP)

## 📌 Overview

This repository contains the implementation of a **Conditional GAN (CGAN)** trained on the **ISIC 2016 dataset**, enhanced with **Wasserstein GAN with Gradient Penalty (WGAN-GP)** to stabilize training and avoid mode collapse.

The project focuses on generating realistic images from sketches by conditioning on class labels, demonstrating the effectiveness of advanced GAN training strategies.

---

## 📂 Dataset

We used the **ISIC 2016 Skin Lesion Dataset** comprising:
- **Training Images**: 9015
- **Training Sketches**: Paired and unpaired
- **Test Images**: 1000
- **Labels**: Provided for both training and test data

---

## 🧠 Methodology

- **Objective**: Generate realistic images from paired/unpaired sketches using a conditional setup.
- **Training Strategy**: WGAN-GP to enforce Lipschitz continuity via gradient penalty.
- **Loss**: Wasserstein distance with gradient penalty term.
- **Stability**: Inner-loop training (multiple Critic updates per Generator update).

---

## ⚙️ Experimental Setup

- **Image Size**: 64 × 64
- **Optimizer**: Adam
- **Loss Functions**: WGAN-GP Loss
- **Metrics**: Frechet Inception Distance (FID), Inception Score (IS)

---

## 🏗 Model Architecture

### Generator
- Input: Sketch + Label (embedded)
- Layers: Embedding → Transposed Convolutions → Upsampling → Tanh Activation

### Discriminator (Critic)
- Input: Image + Label (embedded)
- Layers: Convolutions → Instance Normalization → Leaky ReLU → Final scalar output

---

## 🔁 Training Details

| Hyperparameter | Value       |
|----------------|-------------|
| λ (GP Weight)  | 100         |
| Critic Steps   | 5           |
| Epochs         | 100         |
| Learning Rate  | 1e-4        |

Loss Function:

\[
L = \mathbb{E}_{\tilde{x} \sim P_g}[D(\tilde{x})] - \mathbb{E}_{x \sim P_r}[D(x)] + \lambda \cdot \mathbb{E}_{\hat{x} \sim P_{\hat{x}}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]
\]

---

## 📊 Results

### Image Generation

- Training and Testing images successfully generated for both paired and unpaired sketches.
- Loss curves and performance visualizations plotted via WandB.

### Scores

| Metric              | Value        |
|---------------------|--------------|
| Inception Score (Unfined) | 0.0023 |
| Inception Score (Fine-tuned) | 1.43 |
| FID Score           | 69.90        |

### Classification (Using Fine-Tuned EfficientNet)

| Dataset         | Accuracy (%) |
|-----------------|--------------|
| Train           | 85.07        |
| Validation      | 74.89        |
| Test (Real)     | 63.00        |
| Test (Generated)| 39.41        |

---

## 🔍 Analysis

- WGAN-GP significantly improves training stability over traditional GANs.
- Better performance observed on paired sketches vs. unpaired.
- Inception Score improved with fine-tuning (pretrained models were not well-aligned with ISIC dataset).

---

## ✅ Conclusion

- CGANs allow conditional control over generated samples.
- WGAN-GP helps mitigate non-convergence and mode collapse.
- Further improvements possible by:
  - Increasing epochs
  - Experimenting with alternative architectures
  - Using pre-trained feature extractors tailored to the dataset

---

## 📚 Resources

- [Improved Training of Wasserstein GANs (WGAN-GP)](https://arxiv.org/abs/1704.00028)
- [WGAN Original Paper](https://arxiv.org/abs/1701.07875)
- [WGAN-GP Medium Review](https://sh-tsang.medium.com/brief-review-wgan-gp-improved-training-of-wasserstein-gans-ae3e2acb25b3)

---

## 👨‍💻 Contributors

- **Sahil** (M21MA210)  
- **Aman Kanshotia** (M21MA201)  
- **Sougata Moi** (M23MAC008)
