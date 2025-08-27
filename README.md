# 3D U-Net for Neonatal Mouse Heart Segmentation

[![GitHub repo](https://img.shields.io/badge/GitHub-Project-green?logo=github)](https://github.com/vinaykadam007/3DUnet-neonatalheart-segmentation)  
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)  
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-3D%20U--Net-orange)]()  
[![Microscopy](https://img.shields.io/badge/Light--Sheet-Microscopy-red)]()

---

## 📌 Overview
This project focuses on **3D U-Net–based segmentation** of **cardiac light-sheet fluorescence microscopy (LSFM) images** of neonatal mouse hearts. The pipeline was designed to handle **large volumetric datasets (1000+ slices per stack)**, enabling precise segmentation of multiple cardiac structures for downstream morphometric and functional analysis.

**Objective:** Develop and fine-tune a 3D U-Net deep learning model for robust and accurate segmentation of neonatal mouse cardiac structures from large-scale LSFM datasets.  
**Outcome:** Achieved **0.84 IoU** on validation and test datasets by leveraging preprocessing and **data augmentation** strategies (elastic transforms, flips, grid distortions) to enhance robustness and generalization【50†source】.

---

## 🧪 Methodology

- **Data:** Light-sheet microscopy datasets of neonatal mouse hearts (stacks of 1000+ images).  
- **Preprocessing:**  
  - Resized images to **512×512** pixels.  
  - Applied grayscale normalization.  
  - Augmentation pipeline (horizontal/vertical flips, elastic transforms, grid distortion).  
- **Architecture:** 3D U-Net with encoder–decoder design, convolution layers, pooling, and transpose convolutions for upsampling【50†source】.
- **Training:**  
  - Loss: Dice + Cross-Entropy.  
  - Optimizer: Adam.  
  - Early stopping to prevent overfitting.  
- **Evaluation Metric:** Intersection over Union (IoU).  

---

## 📊 Results

### Without Augmentation
- Validation IoU: **0.49** (poor generalization).  

### With Augmentation
- Validation IoU: **0.84**.  
- Test IoU: ranged from **0.84 to 0.91** across different images【50†source】.

### Sample Predictions
| Input (Raw) | Ground Truth | Prediction |

![Prediction](https://drive.google.com/uc?export=view&id=1k8q5VoDRB4U8M7XD9ePpRx9hff4jjUmV)


**Key Insight:** Data augmentation significantly boosted model generalization and segmentation precision, demonstrating feasibility of applying deep learning to large-scale cardiac imaging datasets【50†source】.

---

## 🙌 Acknowledgements
This work was conducted as part of neonatal cardiac imaging research, integrating **3D deep learning** with **biomedical image analysis**.  

---

✨ If you find this project useful, please ⭐ the repo and share it with the community!
