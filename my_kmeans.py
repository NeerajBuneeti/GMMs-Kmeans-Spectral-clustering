# defining my_kmeans function:
import numpy as np


def my_kmeans(data, K):
    # Initializing cluster centroids by randomly selecting K data points
    count = 0
    np.random.seed(42)
    data = data.to_numpy()
    # choosing random points as centroids in k means.
    centroids_indices = np.random.choice(len(data), K, replace=False)

    centroids = data[centroids_indices]

    while True:
        # Assigning each data point to the nearest centroid
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1) + 1  # Add 1 to shift labels

        # Calculating new centroids as the mean of data points in each cluster
        new_centroids = np.array([data[labels == k].mean(axis=0) for k in range(1, K + 1)])  # Start from 1

        # Checking for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids
        count = count + 1  # the final count will be the number of iterations.
    return labels, count


