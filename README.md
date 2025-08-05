# Graph Convolutional Network (GCN) for Binary Node Classification

This project implements a **Graph Convolutional Network (GCN)** using TensorFlow/Keras to perform binary node classification. The goal is to classify nodes in a graph given a sparse adjacency matrix and an initial embedding representation. This notebook was developed as part of a coursework assignment.

---

## 🧠 Model Overview

The GCN model follows a typical node classification pipeline with:

- Preprocessing Fully-Connected Layer (FFN)
- Two GCN layers for message passing
- Postprocessing FFN
- Output layer with sigmoid activation for binary classification

Model enhancements:
- L2 regularization
- Dropout (commented for compliance with assignment)
- Layer normalization
- Optional weighted loss for imbalanced data
- Stratified train/validation/test split

---

## 📁 Files

- `DL_19_02_2025-3.ipynb` — Jupyter notebook with the full code, model definition, training, and evaluation.
- `input_data.pkl` — Pickled data file containing graph data (features, adjacency matrix, labels).

---

## 📦 Requirements

See [`requirements.txt`](#requirements-txt) or install manually:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
