import seaborn as sns
import matplotlib.pyplot as plt


def perform_full_eda(df):
    """
    Data Health Check.
    Finds nulls, duplicates, and data types to prevent pipeline errors downstream.
    Also prints summary statistics required for the report.
    """
    print("\n" + "=" * 35)
    print("       DATA HEALTH REPORT")
    print("=" * 35)

    # 1. Missing Values
    null_counts = df.isnull().sum()
    print(f"\nMissing Values per Column:\n"
          f"{null_counts[null_counts > 0] if null_counts.any() else '  None found ✓'}")

    # 2. Duplicates
    dupes = df.duplicated().sum()
    print(f"\nDuplicate Rows: {dupes}")

    # 3. Data Types (must all be numeric for distance-based clustering)
    print(f"\nData Types:\n{df.dtypes}")

    # 4. Summary Statistics
    print("\nSummary Statistics:")
    print(df.describe())
    print("=" * 35 + "\n")


def plot_visuals(df):
    """
    Generates all three visual EDA outputs required for the 5-page report:
      1. Correlation Heatmap  — shows feature relationships
      2. Boxplots             — reveals outliers (critical for DBSCAN/K-Means choice)
      3. Pair Plot            — pairwise feature distributions and scatter relationships
    """
    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()

    # 2. Boxplots for Outlier Detection
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df)
    plt.xticks(rotation=45)
    plt.title('Outlier Detection — Boxplots per Feature')
    plt.tight_layout()
    plt.show()

    # 3. Pair Plot — pairwise relationships and distributions
    # diag_kind='kde' shows the distribution shape per feature on the diagonal
    print("Generating Pair Plot (this may take a moment)...")
    pair_grid = sns.pairplot(df, diag_kind='kde', plot_kws={'alpha': 0.4, 's': 20})
    pair_grid.figure.suptitle('Pair Plot — Pairwise Feature Relationships', y=1.02)
    plt.show()
