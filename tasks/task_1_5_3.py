import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def run_task_1_5_3(df):
    # Features and target
    X = df.drop(columns=["quality", "Id"]).values
    y = df["quality"].values

    # Train/test split (just once for plotting)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Multiple regression
    model_multi = LinearRegression().fit(X_train, y_train)
    y_pred_multi = model_multi.predict(X_test)  

    # Simple regression (alcohol only)
    X_alcohol = df[["alcohol"]].values
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X_alcohol, y, test_size=0.2, random_state=42
    )
    model_alcohol = LinearRegression().fit(X_train_a, y_train_a)
    y_pred_alcohol = model_alcohol.predict(X_test_a)

    # Plot Predicted vs Actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_multi, alpha=0.6, label="Multiple Regression")
    plt.scatter(y_test_a, y_pred_alcohol, alpha=0.6, label="Alcohol Only")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=2)  # perfect fit line
    plt.xlabel("Actual Quality")
    plt.ylabel("Predicted Quality")
    plt.title("Predicted vs Actual Wine Quality")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bilder/predicted_vs_actual.png", dpi=300)
    plt.show()