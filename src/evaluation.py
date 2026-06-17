import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage


# ── Internal Metrics ──────────────────────────────────────────────────────────

def get_silhouette(data, labels):
    """
    Silhouette Score: measures how similar a point is to its own cluster
    vs other clusters. Range [-1, 1] — higher is better.
    Returns -1 if fewer than 2 valid clusters exist.
    """
    if len(set(labels)) > 1:
        return silhouette_score(data, labels)
    return -1


def calculate_davies_bouldin(data, labels):
    """
    Davies-Bouldin Index: ratio of within-cluster scatter to between-cluster
    separation. Lower is better. Returns -1 if fewer than 2 clusters.
    """
    if len(set(labels)) > 1:
        return davies_bouldin_score(data, labels)
    return -1


def calculate_calinski_harabasz(data, labels):
    """
    Calinski-Harabasz Index (Variance Ratio Criterion): ratio of between-cluster
    dispersion to within-cluster dispersion. Higher is better.
    Returns -1 if fewer than 2 clusters.
    """
    if len(set(labels)) > 1:
        return calinski_harabasz_score(data, labels)
    return -1


# ── Hyperparameter Tuning Plots ───────────────────────────────────────────────

def plot_elbow_method(data, max_k=10):
    """
    Elbow Plot for K-Means: plots inertia vs k.
    The 'elbow' (point of diminishing returns) suggests the optimal k.
    """
    inertias = []
    K_range = range(1, max_k + 1)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertias, marker='o')
    plt.title('Elbow Method — Optimal k for K-Means')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
    plt.xticks(K_range)
    plt.show()


def plot_k_distance(data, k=5):
    """
    K-Distance Graph for DBSCAN eps selection.
    Plots the distance to each point's k-th nearest neighbour (sorted descending).
    The 'elbow' of the curve is a strong candidate for the eps parameter.
    Returns the automatically detected suggested eps value.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    distances, _ = nbrs.kneighbors(data)

    # Sort distances to k-th neighbour in descending order
    k_distances = np.sort(distances[:, k - 1])[::-1]

    # Auto-detect elbow via maximum curvature (second derivative)
    diffs       = np.diff(k_distances)
    elbow_idx   = np.argmax(np.abs(np.diff(diffs))) + 1
    suggested_eps = k_distances[elbow_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(k_distances, label='k-distance')
    plt.axhline(
        y=suggested_eps,
        color='r',
        linestyle='--',
        label=f'Suggested eps ≈ {suggested_eps:.4f}'
    )
    plt.axvline(x=elbow_idx, color='grey', linestyle=':', alpha=0.6)
    plt.title(f'K-Distance Graph (k={k}) — DBSCAN eps Selection')
    plt.xlabel('Points (sorted by distance)')
    plt.ylabel(f'Distance to {k}th Nearest Neighbour')
    plt.legend()
    plt.show()

    return float(suggested_eps)


def plot_dendrogram(data):
    """
    Dendrogram for Hierarchical Clustering.
    Ward linkage minimises within-cluster variance — a solid default.
    Helps visually confirm the number of clusters.
    """
    linked = linkage(data, method='ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linked, truncate_mode='lastp', p=20, leaf_rotation=45)
    plt.title('Dendrogram — Hierarchical Clustering (Ward Linkage)')
    plt.xlabel('Sample index (or cluster size)')
    plt.ylabel('Distance')
    plt.show()


# ── Cluster Visualisation ─────────────────────────────────────────────────────

def plot_pca_clusters(data, labels, title="PCA Cluster Plot"):
    """
    Scatter plot of PCA-reduced data coloured by cluster label.
    Noise points (label = -1 from DBSCAN) are shown in grey.
    """
    plt.figure(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    palette = sns.color_palette('viridis', n_colors=max(len(unique_labels), 1))
    color_map = {label: ('grey' if label == -1 else palette[i]) for i, label in enumerate(unique_labels)}
    colors = [color_map[l] for l in labels]

    plt.scatter(data[:, 0], data[:, 1], c=colors, s=30, alpha=0.7)
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=color_map[l],
                   markersize=8,
                   label=('Noise' if l == -1 else f'Cluster {l}'))
        for l in unique_labels
    ]
    plt.legend(handles=handles, title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.tight_layout()
    plt.show()


def save_comparison_table_image(results, filename="data/comparison_table.png"):
    """
    Saves the comparison table as a PNG image for easy inclusion in the report.
    """
    df = pd.DataFrame(results).T
    fig, ax = plt.subplots(figsize=(10, 2)) # Adjust size as needed
    ax.axis('off')
    
    # Render the table
    tbl = ax.table(cellText=df.values, 
                   colLabels=df.columns, 
                   rowLabels=df.index, 
                   loc='center', 
                   cellLoc='center')
    
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    plt.title("Algorithm Performance Comparison", pad=20)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Comparison table saved to {filename}")


# ── Comparative Summary ───────────────────────────────────────────────────────

def print_comparison_table(results):
    """
    Prints a formatted comparison table across all algorithms and all metrics.
    results: dict of {algorithm_name: {metric_name: value}}

    Interpretation guide printed below the table:
      Silhouette ↑  — closer to 1 is better
      Davies-Bouldin ↓ — closer to 0 is better
      Calinski-Harabasz ↑ — higher is better
      -1 means fewer than 2 valid clusters (DBSCAN noise dominance)
    """
    df = pd.DataFrame(results).T
    print("\n" + "=" * 55)
    print("       ALGORITHM COMPARISON TABLE")
    print("=" * 55)
    print(df.to_string())
    print("-" * 55)
    print("↑ higher is better  |  ↓ lower is better")
    print("Note: -1 = metric could not be computed (< 2 clusters).")
    print("=" * 55 + "\n")