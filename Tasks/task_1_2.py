import matplotlib.pyplot as plt

def run_task_1_2(df):
    # Q1.2.1 - Correlation matrix
    corr_matrix = df.corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix)

    # Q1.2.2 - Heatmap with values
    fig, ax = plt.subplots(figsize=(12,10))
    cax = ax.matshow(corr_matrix, cmap="coolwarm")
    plt.colorbar(cax)

    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=90)
    ax.set_yticklabels(corr_matrix.columns)

    # Annotate values
    import numpy as np
    for (i, j), val in np.ndenumerate(corr_matrix.values):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=7)

    plt.title("Correlation Matrix Heatmap", fontsize=14)
    plt.tight_layout()
    plt.savefig("Images/correlation_matrix_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Q1.2.3 - Strongest pos/neg correlation with quality
    quality_corr = corr_matrix["quality"].drop("quality")
    strongest_pos = quality_corr.idxmax(), quality_corr.max()
    strongest_neg = quality_corr.idxmin(), quality_corr.min()

    print("\nStrongest positive correlation with quality:", strongest_pos)
    print("Strongest negative correlation with quality:", strongest_neg)

    # Q1.2.4 - Scatter plots for strongest correlations
    
