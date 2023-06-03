import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_excel("null.csv")

# Select relevant features for clustering
features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Perform K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(scaled_features)

# Add cluster labels to the dataset
data['Cluster'] = kmeans.labels_

# Elbow method to determine the optimal number of clusters
wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot the Elbow graph
plt.plot(range(1, 20), wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Graph')
plt.show()

# Visualize the clusters
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('K-Means Clustering - Customer Segmentation')
plt.show()

# Cluster centers (centroids)
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Insights and interpretation
for i in range(len(cluster_centers)):
    print(f"Cluster {i+1}:")
    print(f"Center coordinates: {cluster_centers[i]}")
    print(f"Number of customers in the cluster: {sum(data['Cluster'] == i)}")
    print()
