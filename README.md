# Clustering Techniques Comparison: GMM, K-Means, and Spectral Clustering

Clustering Visualization

## 📌 Overview

This repository explores and compares three powerful clustering techniques:
- **Gaussian Mixture Models (GMMs)**
- **K-Means**
- **Spectral Clustering**

Through hands-on experimentation and evaluation, these techniques are applied to datasets with varying complexities, and their performances are observed using relevant metrics.

## 🧠 Techniques Explored

### 1. Gaussian Mixture Models (GMMs)

GMMs assume data is generated from a mixture of several Gaussian distributions, offering flexibility in modeling non-spherical clusters.

#### Key Points:
- **Cluster Flexibility**: Models ellipsoidal shapes
- **Covariance Matrices**: Captures different cluster shapes
- **Performance Metrics**: Outperformed K-Means in ARI and NMI, especially for non-spherical clusters

### 2. K-Means Clustering

A popular algorithm known for its simplicity and computational efficiency, best suited for spherical and equally sized clusters.

#### Key Points:
- **Simplicity**: Easy to implement, quick convergence
- **Limitations**: Struggles with complex cluster shapes

### 3. Spectral Clustering

A graph-based technique excelling at capturing non-linear structures in data.

#### Key Points:
- **Non-linear Data**: Highly effective for intricate, non-linear shapes
- **Affinity Matrix and Sigma**: Controls cluster formation
- **Eigenvector Transformation**: Allows better K-Means performance in transformed space

## 📊 Comparative Analysis

| Technique | Strengths | Weaknesses | Best Use Case |
|-----------|-----------|------------|---------------|
| GMM | Flexible shapes, Distribution modeling | Computationally intensive | Complex, varied cluster shapes |
| K-Means | Fast, Simple | Limited to spherical clusters | Quick clustering of simple data |
| Spectral | Handles non-linear data | Sensitive to parameter choice | Complex, non-linear distributions |

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- NumPy
- Scikit-learn
- Matplotlib

