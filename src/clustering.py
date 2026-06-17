from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

def run_kmeans(data, k):
    """
    Run K-Means clustering algorithm.
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(data)
    return labels, kmeans

def run_hierarchical(data, n_clusters):
    """
    Run Hierarchical (Agglomerative) clustering algorithm.
    """
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hc.fit_predict(data)
    return labels, hc

def run_dbscan(data, eps, min_samples):
    """
    Run DBSCAN clustering algorithm.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    return labels, dbscan
