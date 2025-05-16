# 📘 CMU 09616: Deep Learning for Scientists - Spring 2025

This repository contains my submissions for the homeworks in CMU 09616 (Deep Learning for Scientists), Spring 2025. Each homework showcases a different neural architecture applied to real-world scientific data problems.

---

## 📁 Homework 1: Neural Network from Scratch

**Goal:** Implement a single-layer MLP from scratch (without autograd) and compare it to a PyTorch baseline on a binary classification task using molecular descriptors.

**Highlights:**
- Built forward and backward passes manually for ReLU, Linear, and SoftmaxCrossEntropy.
- Validated against PyTorch implementation for correctness.
- Dataset based on RDKit molecular descriptors with binary logP labels.

📄 Deliverables:
- `nn.py`: Manual MLP with custom backpropagation
- `reference.py`: PyTorch version
- Training & testing loss/accuracy plots

---

## 🦟 Homework 2: CNNs Against Malaria

**Goal:** Build a Convolutional Neural Network to classify blood cell images as parasitized or uninfected using a dataset of single-cell microscopic images.

**Highlights:**
- Built a custom CNN and also experimented with pretrained models (ResNet, VGG).
- Achieved high mean F1-score on Kaggle leaderboard.
- Visualized feature embeddings using UMAP/t-SNE.

📄 Deliverables:
- Model code and training pipeline
- PDF report with background, architecture details, and performance analysis
- Embedding visualizations

---

## 🧬 Homework 3: Protein Family Classification

**Goal:** Perform multiclass classification of protein sequences using the PFam seed dataset.

**Highlights:**
- Fine-tuned `Rostlab/prot_bert` as the baseline transformer model.
- Improved accuracy by testing other models such as ESM2.
- Used UMAP to visualize protein embeddings and observe clustering by family.

📄 Deliverables:
- `train_baseline.py` and `train_esm.py`: Training scripts
- PDF report with results, visualizations, and methodology
- Final predictions submitted to Kaggle

---

## 🧪 Homework 4: Graph Neural Networks for Molecular Property Prediction

**Goal:** Train a GNN to predict molecular properties using molecular graphs (atoms as nodes, bonds as edges).

**Highlights:**
- Used PyTorch Geometric (`PyG`) to implement GraphConv and GIN-based models.
- Trained on a dataset of molecules with target regression labels.
- Evaluated using Mean Absolute Error (MAE).

📄 Deliverables:
- GNN model and training code using PyG
- Prediction file submitted to Kaggle

---

## 📊 Summary of Techniques

| Homework | Task Type       | Model Type                  | Dataset Modality     | Metric          |
|----------|------------------|-----------------------------|-----------------------|-----------------|
| HW1      | Binary Classification | MLP from scratch           | Tabular (Molecular Descriptors) | Accuracy         |
| HW2      | Binary Classification | CNN / Pretrained CNNs      | Image (Blood Cells)   | Mean F1-Score    |
| HW3      | Multiclass Classification | Transformer (ProteinBERT, ESM) | Sequence (Proteins)   | Accuracy         |
| HW4      | Regression       | GNN (GraphConv, GIN)       | Graph (Molecular Graphs) | MAE             |

---
