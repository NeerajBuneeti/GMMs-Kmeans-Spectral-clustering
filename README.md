

## Overview

In this repository, I explore and compare three powerful clustering techniques—**Gaussian Mixture Models (GMMs)**, **K-Means**, and **Spectral Clustering**. Each of these algorithms has its strengths and limitations, depending on the nature of the data being clustered. Through hands-on experimentation and evaluation, I applied these techniques to datasets with varying complexities and observed their performances using relevant metrics.

Let's dive into a breakdown of my understanding and key takeaways from these methods!

---

### Gaussian Mixture Models (GMMs)
Gaussian Mixture Models are a versatile clustering method that assumes the data is generated from a mixture of several Gaussian distributions. They offer flexibility in modeling clusters that are not necessarily spherical, which gives GMMs a distinct advantage over other techniques like K-Means, especially when dealing with Gaussian-distributed clusters.

#### Key Points:
1. **Cluster Flexibility**: GMM can model ellipsoidal shapes, unlike K-Means, which tends to form spherical clusters.
2. **Covariance Matrices**: By tweaking the covariance matrix, we can capture different cluster shapes, from spherical to diagonal, to unrestricted forms where correlations between features exist.
3. **Performance Metrics**: I used **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)** to evaluate performance. Both metrics indicated GMM outperformed K-Means, especially when clusters had non-spherical shapes.

---

### K-Means Clustering
K-Means is one of the most popular clustering algorithms due to its simplicity and computational efficiency. It works best when the clusters are spherical and equally sized. However, it struggles with complex cluster shapes like elliptical or non-linear distributions.

#### Key Points:
1. **Simplicity**: K-Means is easy to implement and converges quickly, making it a go-to choice for simple, spherical datasets.
2. **Limitations**: When applied to datasets with irregularly shaped clusters (like elliptical Gaussians), K-Means tends to misclassify points. Its centroids cannot capture correlation between features as efficiently as GMM.

---

### Spectral Clustering
Spectral Clustering is a graph-based clustering technique that excels at capturing non-linear structures in the data. It constructs a similarity graph, transforms it using a Laplacian matrix, and applies K-Means to the transformed data.

#### Key Points:
1. **Non-linear Data**: Spectral Clustering is highly effective in datasets where clusters form intricate, non-linear shapes. For instance, it easily handles spiral-shaped clusters, which K-Means struggles with
2. **Affinity Matrix and Sigma**: The similarity graph is constructed using Gaussian kernels. The parameter **sigma** controls the spread of the kernel, affecting how data points are clustered. Small sigma values only consider nearby points, while larger values encompass more distant points.
3. **Eigenvector Transformation**: Spectral Clustering transforms the data into a new space defined by the eigenvectors of the Laplacian matrix, allowing K-Means to work better in this transformed space.

---

### Conclusion: When to Use Each Method?
- **GMMs** are best when you expect clusters of different shapes and want to model their distributions flexibly.
- **K-Means** works well for quick clustering of simple, spherical data.
- **Spectral Clustering** is ideal for complex, non-linear data distributions that cannot be handled by traditional methods like K-Means.

---

### Let's Collaborate!

I am passionate about Machine Learning and Artificial Intelligence, and always open to collaborating on exciting projects. If you're interested in exploring new ideas together, feel free to reach out to my mail: voonadhanvanth183@gmail.com

Happy coding and clustering! :)
