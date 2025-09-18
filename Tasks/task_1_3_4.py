import numpy as np
import matplotlib.pyplot as plt

def plot_regression(df, feature: str, w: float, b: float, filename: str):
    """
    Plot regression line against data points for a given feature.
    """
    # Extract X and y
    X = df[feature].values
    y = df["quality"].values

    # Standardize X (if your model was trained on standardized features)
    X_std = (X - X.mean()) / X.std()

    # Predictions
    y_pred = w * X_std + b

    # Plot
    plt.scatter(X, y, label="Data", alpha=0.5)
    plt.plot(X, y_pred, color="red", label="Regression line")
    plt.xlabel(feature.capitalize())
    plt.ylabel("Quality")
    plt.title(f"Regression fit: Quality vs. {feature.capitalize()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
