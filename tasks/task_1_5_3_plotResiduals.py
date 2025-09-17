import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def plot_residuals(df):
    # Multiple regression
    X = df.drop(columns=["quality", "Id"]).values
    y = df["quality"].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Quality")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residuals Plot (Multiple Regression)")
    plt.tight_layout()
    plt.savefig("bilder/residuals_plot.png", dpi=300)
    plt.show()
