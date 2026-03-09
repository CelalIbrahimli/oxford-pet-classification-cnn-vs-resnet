# Oxford-IIIT Pet Breed Classification (CNN vs ResNet18)

This project explores **image classification using deep learning in PyTorch** on the **Oxford-IIIT Pet dataset**, which contains **37 cat and dog breeds**.

The goal of the project is to compare:

- A **custom Convolutional Neural Network (CNN)** trained from scratch
- A **transfer learning approach using ResNet18 pretrained on ImageNet**

This comparison demonstrates the impact of pretrained feature extractors on performance when working with relatively small datasets.

---

# Dataset

Oxford-IIIT Pet Dataset

- 37 breed categories (cats and dogs)
- ~7,300 images
- High intra-class similarity and subtle visual differences between breeds

Dataset source:

https://www.robots.ox.ac.uk/~vgg/data/pets/

In this project the dataset is loaded directly using **torchvision.datasets.OxfordIIITPet**.

---

# Project Pipeline

1. Dataset loading using torchvision
2. Safe image loading using a custom `SafeImageDataset`
3. Data preprocessing and normalization
4. Training a custom CNN baseline
5. Training a ResNet18 model using transfer learning
6. Performance comparison
7. Visualization of model predictions

---

# Model Architectures

## Custom CNN (Baseline)

Architecture:

Conv → BatchNorm → ReLU → MaxPool  
Conv → BatchNorm → ReLU → MaxPool  
Conv → BatchNorm → ReLU → MaxPool  
Conv → BatchNorm → ReLU → MaxPool  
AdaptiveAvgPool → Fully Connected Layers

This model is trained **from scratch**.

---

## ResNet18 (Transfer Learning)

ResNet18 pretrained on **ImageNet** is used.

Strategy:

- Pretrained backbone
- Backbone **frozen**
- Only the final classification layer is trained

This allows the model to leverage strong pretrained visual features.

---

# Results

| Model | Training Type | Epochs | Best Accuracy |
|------|---------------|-------|---------------|
| Custom CNN | Trained from scratch | 10 | ~11.7% |
| ResNet18 | Transfer Learning | 10 | **87.68%** |

---

# Key Insights

- Training a CNN from scratch on a small dataset is difficult.
- Transfer learning dramatically improves performance.
- Pretrained CNN backbones provide strong general visual features that transfer well to new tasks.

---

# Sample Predictions

The notebook includes visualization of model predictions with:

- Correct predictions highlighted in **green**
- Incorrect predictions highlighted in **red**

Example output:
