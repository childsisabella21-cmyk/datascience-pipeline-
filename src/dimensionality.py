import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def apply_pca(data, n_components=2):
    """
    Standard PCA execution. 
    Added a check to ensure we don't ask for more components than features.
    """
    max_components = min(data.shape[0], data.shape[1])
    if n_components > max_components:
        n_components = max_components
        
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    return transformed_data, pca.explained_variance_ratio_

def plot_scree_plot(data):
    """
    This shows the Elbow for PCA. It helps justify why we chose 2 or 3 components.
    """
    pca = PCA().fit(data)
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Scree Plot: Explained Variance by Components')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    plt.legend()
    plt.show()