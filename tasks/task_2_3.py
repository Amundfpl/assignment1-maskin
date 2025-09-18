from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
import numpy as np

def run_task_2_3(df, alpha=1.0):
    X = df.drop(columns=["quality", "Id"]).values
    y = df["quality"].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results_ridge, results_lasso = [], []

    print("Evaluating Ridge and Lasso Regression (alpha={})".format(alpha))

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Ridge
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)
        mse_r = np.mean((y_test - y_pred_ridge)**2)
        rmse_r = np.sqrt(mse_r)
        ss_res = np.sum((y_test - y_pred_ridge)**2)
        ss_tot = np.sum((y_test - np.mean(y_test))**2)
        r2_r = 1 - ss_res/ss_tot
        results_ridge.append((mse_r, rmse_r, r2_r))

        # Lasso
        lasso = Lasso(alpha=alpha, max_iter=5000)
        lasso.fit(X_train, y_train)
        y_pred_lasso = lasso.predict(X_test)
        mse_l = np.mean((y_test - y_pred_lasso)**2)
        rmse_l = np.sqrt(mse_l)
        ss_res = np.sum((y_test - y_pred_lasso)**2)
        ss_tot = np.sum((y_test - np.mean(y_test))**2)
        r2_l = 1 - ss_res/ss_tot
        results_lasso.append((mse_l, rmse_l, r2_l))

        print(f"Fold {fold}:")
        print(f"  Ridge  -> MSE={mse_r:.4f}, RMSE={rmse_r:.4f}, R2={r2_r:.4f}")
        print(f"  Lasso  -> MSE={mse_l:.4f}, RMSE={rmse_l:.4f}, R2={r2_l:.4f}")

    # Averages
    ridge_mean = np.mean(results_ridge, axis=0)
    lasso_mean = np.mean(results_lasso, axis=0)

    print("\nAverage across folds:")
    print(f"  Ridge  -> MSE={ridge_mean[0]:.4f}, RMSE={ridge_mean[1]:.4f}, R2={ridge_mean[2]:.4f}")
    print(f"  Lasso  -> MSE={lasso_mean[0]:.4f}, RMSE={lasso_mean[1]:.4f}, R2={lasso_mean[2]:.4f}")

    return ridge_mean, lasso_mean, ridge.coef_, lasso.coef_
