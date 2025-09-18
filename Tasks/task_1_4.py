import numpy as np
from sklearn.model_selection import KFold

def gradient_descent(X, y, alpha=0.01, epochs=1000):
    """Simple linear regression with batch gradient descent."""
    w, b = 0.0, 0.0
    n = len(X)

    for _ in range(epochs):
        y_pred = w * X + b
        error = y - y_pred

        dw = -(2/n) * np.dot(X, error)
        db = -(2/n) * np.sum(error)

        w -= alpha * dw
        b -= alpha * db

    return w, b

def evaluate_model(y_true, y_pred):
    """Compute MSE, RMSE, R² score."""
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    return mse, rmse, r2

def run_task_1_4(df, feature="alcohol", k=5):
    """
    Perform k-fold cross-validation with simple linear regression
    predicting wine quality from the given feature.
    
    Parameters:
        df (DataFrame): dataset
        feature (str): column name (e.g. "alcohol" or "chlorides")
        k (int): number of folds (default 5)
    """
    X = df[feature].values
    y = df["quality"].values

    # Standardize predictor
    X = (X - X.mean()) / X.std()

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    print(f"\nEvaluating predictor: {feature}\n")

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train model
        w, b = gradient_descent(X_train, y_train)

        # Predictions
        y_pred = w * X_test + b

        # Evaluate
        mse, rmse, r2 = evaluate_model(y_test, y_pred)
        fold_results.append((mse, rmse, r2))

        print(f"Fold {fold}: MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    # Compute averages + variances
    mse_vals = [r[0] for r in fold_results]
    rmse_vals = [r[1] for r in fold_results]
    r2_vals = [r[2] for r in fold_results]

    print("\nAverage across folds:")
    print(f"MSE={np.mean(mse_vals):.4f}, Var={np.var(mse_vals):.4f}")
    print(f"RMSE={np.mean(rmse_vals):.4f}, Var={np.var(rmse_vals):.4f}")
    print(f"R2={np.mean(r2_vals):.4f}, Var={np.var(r2_vals):.4f}")

    return fold_results
