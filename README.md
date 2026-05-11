# Humpback Whale Individual Identification

This project builds machine learning models to identify **individual humpback whales** from tail‑fluke images using a filtered subset of the Kaggle Humpback Whale Identification dataset.

## Overview
We compare three approaches:
- **PCA + SVM**
- **Keras ANN**
- **Custom CNN**

The goal is to evaluate how different algorithms perform on a **fine‑grained visual recognition task** where subtle texture and shape differences distinguish one whale from another.

## Dataset
- Source: Kaggle Humpback Whale Identification
- Removed all `new_whale` labels
- Kept **top 20 whale IDs** with ≥20 images each
- Images resized to **96×96 grayscale**
- Final size: **~600–700 images**, stratified 80/20 split

## Methods
### PCA + SVM
- PCA → 300 components  
- RBF‑kernel SVM with GridSearchCV  
- **Accuracy:** 55–70%

### Keras ANN
- Dense 512 → Dense 128 → Softmax(20)  
- GridSearchCV for batch size + epochs  
- **Accuracy:** 45–65%

### Custom CNN
- 4 Conv blocks (32→256 filters), BatchNorm, MaxPool, Dropout  
- Dense 512 + Dropout  
- EarlyStopping + LR scheduling  
- **Accuracy:** 65–80%

## Key Findings
- **CNN performed best**, capturing spatial fluke patterns  
- **PCA+SVM** was a strong, efficient baseline  
- Dataset size and imbalance were the main limitations  

## Team
- **Lisa Liu** — Model Training, Documentation  
- **Osvaldo Valdiviezo** — Data Analysis, CNN Training, Documentation  
- **Kenia Sanchez‑Macario** — Data Analysis, Documentation, Presentation
