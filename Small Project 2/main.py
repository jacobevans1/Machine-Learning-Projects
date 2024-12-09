import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset
data = pd.read_csv('wine_no_label.csv')
scaler = StandardScaler()
X = scaler.fit_transform(data[['Alcohol','Malic.acid']].values)


# Elbow method for KMeans
def plot_elbow(X):
    distortions = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
        km.fit(X)
        distortions.append(km.inertia_)

    plt.figure()
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for KMeans')
    plt.tight_layout()
    plt.show()


# Scatter plot of clusters
def plot_clusters(X, labels, cluster_centers=None, title="Clusters"):
    plt.figure()
    unique_labels = np.unique(labels)
    for label in unique_labels:
        plt.scatter(X[labels == label, 0], X[labels == label, 1], s=50, label=f"Cluster {label + 1}")
    if cluster_centers is not None:
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=250, marker='*', c='red', edgecolor='black',
                    label='Centroids')
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.tight_layout()
    plt.show()


# Silhouette plot
def plot_silhouette(X, labels, title):
    n_clusters = len(np.unique(labels))
    silhouette_vals = silhouette_samples(X, labels)
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []

    plt.figure()
    for i in range(n_clusters):
        c_silhouette_vals = silhouette_vals[labels == i]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0)
        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--")
    plt.yticks(yticks, [f"Cluster {i + 1}" for i in range(n_clusters)])
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.title(f'Silhouette analysis for {title}')
    plt.tight_layout()
    plt.show()


# Apply KMeans clustering
def apply_kmeans(X, n_clusters):
    km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=0)
    labels = km.fit_predict(X)
    plot_clusters(X, labels, km.cluster_centers_, title=f"KMeans {n_clusters} Clusters")
    plot_silhouette(X, labels, f"KMeans {n_clusters} Clusters")


# Apply Hierarchical clustering
def apply_hierarchical(X, n_clusters):
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hc.fit_predict(X)
    plot_clusters(X, labels, title=f"Hierarchical {n_clusters} Clusters")
    plot_silhouette(X, labels, f"Hierarchical {n_clusters} Clusters")


# Apply DBSCAN
def apply_dbscan(X, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    plot_clusters(X, labels, title="DBSCAN")
    plot_silhouette(X, labels, "DBSCAN")


# Plot the elbow method
plot_elbow(X)

# Apply KMeans for 3, 4, 5 clusters
apply_kmeans(X, 3)
apply_kmeans(X, 4)
apply_kmeans(X, 5)

# Apply Hierarchical clustering for 3, 4, 5 clusters
apply_hierarchical(X, 3)
apply_hierarchical(X, 4)
apply_hierarchical(X, 5)

# Apply DBSCAN with chosen parameters
apply_dbscan(X, eps=0.5, min_samples=5)