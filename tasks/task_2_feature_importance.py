import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_task_2_1(df):
    # Features and target
    X = df.drop(columns=["quality", "Id"]).values
    y = df["quality"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features (fit only on train, apply to both train & test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train multiple regression
    model_multi = LinearRegression().fit(X_train_scaled, y_train)
    y_pred_multi = model_multi.predict(X_test_scaled)

    # Feature importance = absolute value of coefficients
    feature_importance = np.abs(model_multi.coef_)
    feature_names = df.drop(columns=["quality", "Id"]).columns
    sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_features = feature_names[sorted_idx]
    sorted_importance = feature_importance[sorted_idx]

    # Print ranking
    print("Feature Importance (sorted):")
    for feature, importance in zip(sorted_features, sorted_importance):
        print(f"{feature}: {importance:.4f}")

    # Plot ranking
    plt.figure(figsize=(8, 6))
    plt.barh(sorted_features, sorted_importance, color="skyblue")
    plt.xlabel("Absolute Coefficient (Importance)")
    plt.title("Feature Importance in Predicting Wine Quality")
    plt.tight_layout()
    plt.savefig("bilder/feature_importance.png", dpi=300)
    plt.show()
