# my_spectralclustering.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from scipy.sparse.linalg import eigs

def my_spectralclustering(data, K, sigma):
    # Calculate the pairwise distance matrix based on Gaussian kernel
    pairwise_dist = pairwise_distances(data, metric='euclidean', squared=True)
    aff_matrix = np.exp(-pairwise_dist / (2 * sigma**2))  #affinity matrix

    # Creating a graph Laplacian matrix
    degree_matrix = np.diag(np.sum(aff_matrix, axis=1))
    laplacian_matrix = degree_matrix - aff_matrix  #(D-A)

    # Computing the first K eigenvectors of the Laplacian matrix
    _, eigenvectors = eigs(laplacian_matrix, k=K, which='SM')
    # Using only the real part of the eigenvector matrix
    eigenvectors = np.real(eigenvectors)
    # Normalizing the rows of the eigenvector matrix
    normalized_eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=1)[:, np.newaxis]

    # Performing K-means clustering on the normalized eigenvectors
    km = KMeans(n_clusters=K, random_state=42)
    cluster_labels = km.fit_predict(normalized_eigenvectors) + 1  # added 1 to start cluster labels from 1.

    return cluster_labels

