import src.initialization as init
import src.data_loader as loader
import src.eda as eda
import src.preprocessing as preprocessor
import src.dimensionality as dim
import src.clustering as cluster
import src.evaluation as eval


def main():
    # 0. Initialization
    init.setup_project_structure()

    # 1. Load Data (path passed as variable — no hard-coded paths)
    df = loader.load_data("data/wholesale_customers.csv")
    if df is None:
        print("Data could not be loaded. Please download wholesale_customers.csv into the data/ folder.")
        return

    # 1.5 EDA — Data health report + all required visuals
    eda.perform_full_eda(df)
    eda.plot_visuals(df)

    # 2. Preprocessing Pipeline (impute → scale)
    pipeline = preprocessor.get_pipeline()
    X_scaled = pipeline.fit_transform(df)

    # 3. Dimensionality Reduction — PCA
    print("\n--- DIMENSIONALITY REDUCTION (PCA) ---")
    print("Generating Scree Plot to justify n_components (close window to continue)...")
    dim.plot_scree_plot(X_scaled)
    X_pca, variance = dim.apply_pca(X_scaled, n_components=3)
    print(f"Variance explained by 3 PCA components: {variance.sum():.2%}")

    # 4. Hyperparameter Tuning (required for all 3 algorithms)
    print("\n--- HYPERPARAMETER TUNING ---")

    print("Generating Elbow Plot for K-Means k selection (close window to continue)...")
    eval.plot_elbow_method(X_pca, max_k=10)

    print("Generating Dendrogram for Hierarchical Clustering (close window to continue)...")
    eval.plot_dendrogram(X_pca)

    print("Generating K-Distance Graph for DBSCAN eps selection (close window to continue)...")
    recommended_eps = eval.plot_k_distance(X_pca, k=5)
    print(f"Suggested eps from K-Distance Graph: {recommended_eps:.4f}")

    # 5. Run All Three Algorithms
    print("\n--- CLUSTERING ---")

    # K-Means — k chosen from elbow plot above
    K = 5
    km_labels, _ = cluster.run_kmeans(X_pca, k=K)
    print(f"K-Means done (k={K}).")
    eval.plot_pca_clusters(X_pca, km_labels, title=f"K-Means Clusters (k={K})")

    # Hierarchical (Agglomerative) — same n_clusters for fair comparison
    hc_labels, _ = cluster.run_hierarchical(X_pca, n_clusters=K)
    print(f"Hierarchical Clustering done (n_clusters={K}).")
    eval.plot_pca_clusters(X_pca, hc_labels, title=f"Hierarchical Clusters (n={K})")

    # DBSCAN — eps from k-distance graph, min_samples=5 (rule of thumb: 2*n_features)
    db_labels, _ = cluster.run_dbscan(X_pca, eps=0.8, min_samples=5)
    n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = list(db_labels).count(-1)
    print(f"DBSCAN done. Clusters found: {n_db_clusters} | Noise points: {n_noise}.")
    eval.plot_pca_clusters(X_pca, db_labels, title="DBSCAN Clusters")

    # 6. Evaluate All Three Algorithms on All Three Mandatory Metrics
    print("\n--- EVALUATION ---")
    results = {}

    for name, labels in [("K-Means", km_labels), ("Hierarchical", hc_labels), ("DBSCAN", db_labels)]:
        sil   = eval.get_silhouette(X_pca, labels)
        dbi   = eval.calculate_davies_bouldin(X_pca, labels)
        chi   = eval.calculate_calinski_harabasz(X_pca, labels)
        results[name] = {
            "Silhouette ↑":         round(sil, 4),
            "Davies-Bouldin ↓":     round(dbi, 4),
            "Calinski-Harabasz ↑":  round(chi, 2),
        }
    #7. Save the comparison table as an image
    eval.save_comparison_table_image(results)
    # 8. Comparative Summary Table
    eval.print_comparison_table(results)


if __name__ == "__main__":
    main()