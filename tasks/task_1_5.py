import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(y_true, y_pred):
    """Compute MSE, RMSE, R2 score."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, r2

def run_task_1_5(df):
    """
    Train a multiple linear regression model using all features to predict quality.
    Use the same 5-fold splits as before and evaluate with MSE, RMSE, and R2.
    """
    # Use all features except quality and Id
    X = df.drop(columns=["quality", "Id"]).values
    y = df["quality"].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train multiple linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        mse, rmse, r2 = evaluate_model(y_test, y_pred)
        fold_results.append((mse, rmse, r2))
        print(f"Fold {fold}: MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    # Compute averages
    results = np.array(fold_results)
    mean_results = results.mean(axis=0)
    print("\nAverage across folds:")
    print(f"MSE={mean_results[0]:.4f}, RMSE={mean_results[1]:.4f}, R2={mean_results[2]:.4f}")

    return fold_results, mean_results