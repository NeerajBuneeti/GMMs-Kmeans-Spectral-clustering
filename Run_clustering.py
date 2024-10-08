#Lets understand the data before applying clustering algorithms

import scipy.io
from my_kmeans import my_kmeans
from my_spectralclustering import my_spectralclustering
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Loading the 7.m files
m1 = scipy.io.loadmat('D:/Projects/assignments/toydata/cluster/data_Aggregation')
m2 = scipy.io.loadmat('D:/Projects/assignments/toydata/cluster/data_Bridge')
m3 = scipy.io.loadmat('D:/Projects/assignments/toydata/cluster/data_Compound')
m4 = scipy.io.loadmat('D:/Projects/assignments/toydata/cluster/data_Flame')
m5 = scipy.io.loadmat('D:/Projects/assignments/toydata/cluster/data_Jain')
m6 = scipy.io.loadmat('D:/Projects/assignments/toydata/cluster/data_Spiral')
m7 = scipy.io.loadmat('D:/Projects/assignments/toydata/cluster/data_TwoDiamonds')

#finding out the column headers
print("The column headers are:")
print(m1.keys(), "\n", m2.keys(),"\n",m3.keys(),"\n", m4.keys(),
    "\n", m5.keys(), "\n", m6.keys(), "\n", m7.keys())

# lets transform our data from m files into dataframe

#from the question we know ”D” is the data matrix and ”L” is
#the ground truth label matrix. So we will save d in x variable,
# and l in y variable(target
#or label)
#Labels:
y_m1, y_m2, y_m3, y_m4, y_m5, y_m6, y_m7 = m1['L'], m2['L'], m3['L'], m4['L'], m5['L'], m6['L'], m7['L']

#Data points:
x_m1, x_m2, x_m3, x_m4, x_m5, x_m6, x_m7 = m1['D'], m2['D'], m3['D'], m4['D'], m5['D'], m6['D'], m7['D']

#Lets print the data together for one cluster file
print("Data in m1 file:", "\n")
print(x_m1, y_m1)


#---------y[labels]---------------
y_m1, y_m2, y_m3, y_m4, y_m5, y_m6, y_m7 = pd.DataFrame(y_m1), pd.DataFrame(y_m2), pd.DataFrame(y_m3), pd.DataFrame(y_m4), pd.DataFrame(y_m5), pd.DataFrame(y_m6), pd.DataFrame(y_m7)
#------------x[Data points]------------------------------
x_m1, x_m2, x_m3, x_m4, x_m5, x_m6, x_m7 = pd.DataFrame(x_m1), pd.DataFrame(x_m2), pd.DataFrame(x_m3), pd.DataFrame(x_m4), pd.DataFrame(x_m5), pd.DataFrame(x_m6), pd.DataFrame(x_m7)


#take a preview of the dataset
print("cluster 1 data \n", x_m1.head())
print(x_m1.info(), "\n", y_m1.info())


print(y_m1[0].unique(),  "\n", y_m2[0].unique(),  "\n", y_m3[0].unique(),
      "\n", y_m4[0].unique() ,  "\n", y_m5[0].unique(),  "\n", y_m6[0].unique(),
      "\n", y_m7[0].unique() )

# Defining function to perform the k-means on all the  7 datasets, instead of rewriting every time
# the arguments for function will be input data(for ex: x_m1), labels(for ex: y_m1), k and my_kmeans.
def kmeans_clustering(data, true_labels, K, kmeans_clus):
    # Calling the provided kmeans function
    cluster_labels, iteration = kmeans_clus(data, K)
    # Creating a DataFrame for visualization
    df = pd.DataFrame(data)
    df['true_labels'] = true_labels
    df['cluster_labels'] = cluster_labels
    # Scatter plot for actual clusters
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(df[0], df[1], c=df['true_labels'], cmap='viridis')
    plt.title('Actual Clusters')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    # Scatter plot for predicted clusters (K-means)
    plt.subplot(1, 2, 2)
    plt.scatter(df[0], df[1], c=df['cluster_labels'], cmap='viridis')
    plt.title('Predicted Clusters (K-means)')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.tight_layout()
    plt.show()
    # Print the cluster labels and the number of iterations
    print(cluster_labels)
    print("Number of iterations:", iteration)

# (a) Aggregation (K=7)
K = 7
kmeans_clustering(x_m1, y_m1, K, my_kmeans)
# (b) Bridge (K=2)
K = 2
kmeans_clustering(x_m2, y_m2, K, my_kmeans)
# (c) Compound (K=6)
K = 6
kmeans_clustering(x_m3, y_m3, K, my_kmeans)
# (d) Flame (K=2)
K = 2
kmeans_clustering(x_m4, y_m4, K, my_kmeans)
# (e) Jain (K=2)
K = 2
kmeans_clustering(x_m5, y_m5, K, my_kmeans)
# (f) Spiral (K=3)
K = 3
kmeans_clustering(x_m6, y_m6, K, my_kmeans)
# (g) TwoDiamond (K=2)
K = 2
kmeans_clustering(x_m7, y_m7, K, my_kmeans)

# I would have used same kmeans function to plot spectral clustering,
# but since we have additonal argument sigma, to avoid confusion
# I am defining a new function.
def spectral_clustering(data, true_labels, K, spect_clus, sigma):
    cluster_labels = spect_clus(data, K, sigma)
    # Creating a DataFrame for visualization
    df = pd.DataFrame(data)
    df['true_labels'] = true_labels
    df['cluster_labels'] = cluster_labels
    # Scatter plot for actual clusters
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(df[0], df[1], c=df['true_labels'], cmap='viridis')
    plt.title('Actual Clusters')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    # Scatter plot for predicted clusters (K-means)
    plt.subplot(1, 2, 2)
    plt.scatter(df[0], df[1], c=df['cluster_labels'], cmap='viridis')
    plt.title(f'Predicted Clusters (Spectral_clustering, sigma = {sigma})')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.tight_layout()
    plt.show()
    # Print the cluster labels and the number of iterations
    print(cluster_labels)


#lets evaluate the clustering by varying the sigma values.
# (g) Aggregation (K=2)
#case 1
K = 2
s = 1
spectral_clustering(x_m7, y_m7, K, my_spectralclustering, s )
#case 2

K = 2
s = 1.5
spectral_clustering(x_m7, y_m7, K, my_spectralclustering, s )
#case 3
K = 2
s = 15
spectral_clustering(x_m7, y_m7, K, my_spectralclustering, s )
# (f)
#case 1
K = 3
s = 1
spectral_clustering(x_m6, y_m6, K, my_spectralclustering, s )
#case 2

K = 3
s = 1.5
spectral_clustering(x_m6, y_m6, K, my_spectralclustering, s )
#case 3

K = 3
s = 15
spectral_clustering(x_m6, y_m6, K, my_spectralclustering, s )
# (c) Aggregation (K=6)
#case 1
K = 6
s = 1
spectral_clustering(x_m3, y_m3, K, my_spectralclustering, s )
#case 2

K = 6
s = 1.5
spectral_clustering(x_m3, y_m3, K, my_spectralclustering, s )
#case 3

K = 6
s = 15
spectral_clustering(x_m3, y_m3, K, my_spectralclustering, s )

# (a) Aggregation (K=7)
#case 1
K = 7
s = 1
spectral_clustering(x_m1, y_m1, K, my_spectralclustering, s )
#case 2

K = 7
s = 10
spectral_clustering(x_m1, y_m1, K, my_spectralclustering, s)

#case 3

K = 7
s = 25
spectral_clustering(x_m1, y_m1, K, my_spectralclustering, s)

# (c) Aggregation (K=6)
#case 1
K = 6
s = 1
spectral_clustering(x_m3, y_m3, K, my_spectralclustering, s )
#case 2

K = 6
s = 1.5
spectral_clustering(x_m3, y_m3, K, my_spectralclustering, s )
#case 3

K = 6
s = 15
spectral_clustering(x_m3, y_m3, K, my_spectralclustering, s )
#e)
K = 2
s = 1
spectral_clustering(x_m5, y_m5, K, my_spectralclustering, s )