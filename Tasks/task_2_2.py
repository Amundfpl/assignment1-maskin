from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def run_task_2_2(df, degree=2):
    # Features and target
    X = df.drop(columns=["quality", "Id"]).values
    y = df["quality"].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    # Polynomial regression pipeline (adds polynomial features + linear regression)
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linreg", LinearRegression())
    ])

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        fold_results.append((mse, rmse, r2))

        print(f"Fold {fold}: MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    results = np.array(fold_results)
    mean_results = results.mean(axis=0)

    print("\nAverage across folds:")
    print(f"MSE={mean_results[0]:.4f}, RMSE={mean_results[1]:.4f}, R2={mean_results[2]:.4f}")

    return fold_results, mean_results
