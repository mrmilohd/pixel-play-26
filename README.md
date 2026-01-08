# PixelPlay ’26 — Video Anomaly Detection

This repository contains our solution for **PixelPlay ’26**, focusing on **unsupervised video anomaly detection**. The pipeline handles corrupted inputs and detects abnormal events using multiple reconstruction- and feature-based models.

---

## Flip Detection CNN (Pre-processing Step)

Some test videos contained **vertically flipped frames**, which caused incorrect anomaly scores.  
A small CNN is used **before all models** to detect and fix this issue.

- **Input:** Single video frames  
- **Training:** Synthetic dataset where 50% of normal frames are vertically flipped  
- **Model:**
  - 3 × Conv2D + ReLU  
  - Adaptive Average Pooling  
  - Sigmoid output  
- **Inference:** Frames predicted as flipped (above threshold) are inverted and saved  
- **Effect:** Reduces reconstruction errors and stabilizes anomaly scores across all methods  

---

## 1. Spatio-Temporal Convolutional Autoencoder *(Best Performing)*

A reconstruction-based model that learns normal **appearance and motion** from short video clips.

- **Input:** 16 consecutive frames stacked as spatio-temporal cuboids  
- **Architecture:**
  - Encoder: 4 convolution blocks (64 → 128 → 256 → 512)  
  - Decoder: Symmetric transposed convolutions  
- **Temporal Handling:** Sliding window, stride = 2  
- **Loss:** Mean Squared Error (MSE)  
- **Anomaly Score:** Reconstruction error  
- **Post-processing:** Global normalization + temporal smoothing  

### Result
- **Best score:** **0.66881**
- Achieved using **60% clean / 40% noisy data augmentation**
- **Highest leaderboard score among all approaches**

---

## 2. VideoPatchCore (CLIP-Based Feature Matching)

A memory-based anomaly detection method using **frozen CLIP features** at spatial, temporal, and global levels.

### Input
- **Clip Length:** 10 frames  
- **Global Stream:** Full frames resized to 224×224  
- **Local Stream:** Person-centric tubes extracted using cached YOLOv5l detections  

### Feature Extraction
- **Backbone:** CLIP RN101 (frozen)  
- **Layers Used:** layer2[-1], layer3[-1]  

### Feature Types
- **Spatial:** Channel-pooled local features (CNL_POOL = 32)  
- **Temporal:** Frame-difference based motion features  
- **Global:** High-level semantic features using temporal pyramid pooling  

### Data Augmentation
- Uniform noise in range `[-0.1373, 0.1373]` with 40% probability  
- Standard CLIP mean/std normalization  

### Memory Banks (after coreset selection)
- Spatial: 3000 samples  
- Temporal: 2000 samples  
- Global: 1000 samples  

### Scoring
- **Distance:** L2 distance to nearest memory sample  
- **Fusion (POWER_N = 4):**
  - Local = `0.7 × Spatial + 0.3 × Temporal`
  - Final = `0.7 × Local + 0.3 × Global`
- **Post-processing:** Min-max normalization over the test set  

---

## 3. Convolutional Autoencoder (Reconstruction-Based)

A deep autoencoder trained to reconstruct **normal spatio-temporal patterns**.

### Input
- 16 grayscale frames resized to `128 × 128`  
- Input shape: `(B, 16, 128, 128)`  
- Temporal stride = 2  

### Architecture

**Encoder**
- Conv(16→64) → BN → ReLU → Dropout(0.2)  
- Conv(64→128) → BN → ReLU → Dropout(0.2)  
- Conv(128→256) → BN → ReLU → Dropout(0.2)  
- Bottleneck: Conv(256→512) → BN → ReLU  

**Decoder**
- ConvT(512→256) → BN → ReLU  
- ConvT(256→128) → BN → ReLU  
- ConvT(128→64) → BN → ReLU  
- ConvT(64→16) → Sigmoid  

### Training Augmentation
- Best performance with **60% clean / 40% noisy** samples  
- Noise: Uniform `[-0.1373, 0.1373]`, clamped to `[0, 1]`  
- Vertical flipping supported but disabled  

### Inference
- **Score:** Per-frame MSE between input and reconstruction  
- **Normalization:** Global min-max over test set  
- **Smoothing:** Moving average (window = 5)  

---

## 4. Denoising Convolutional Autoencoder

A noise-robust version of the ConvAutoencoder.

- **Architecture & Input:** Same as the standard autoencoder  
- **Training:** Model learns to reconstruct clean frames from noisy inputs  
- **Noise:** Uniform symmetric noise (≈ 0.157 intensity)  
- **Inference:** No noise added; abnormal events produce higher errors  
- **Post-processing:**
  - Normalization using 1st–100th percentiles  
  - Larger smoothing window (15)  

---

## Summary

- Flip detection fixes corrupted inputs and improves all models  
- Spatio-temporal autoencoder achieved the **best score (0.66881)**  
- VideoPatchCore captures spatial, motion, and semantic anomalies  
- Denoising autoencoder improves robustness to noisy inputs  
