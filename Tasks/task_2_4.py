from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import numpy as np

def run_task_2_4(df):
    # Features and target
    X = df.drop(columns=["quality", "Id"]).values
    y = df["quality"].values
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results_linear, results_tree, results_rf = [], [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Linear regression baseline
        linreg = LinearRegression().fit(X_train, y_train)
        y_pred_lin = linreg.predict(X_test)
        mse = np.mean((y_test - y_pred_lin)**2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((y_test - y_pred_lin)**2) / np.sum((y_test - np.mean(y_test))**2)
        results_linear.append((mse, rmse, r2))

        # Decision Tree
        tree = DecisionTreeRegressor(max_depth=5, random_state=42).fit(X_train, y_train)
        y_pred_tree = tree.predict(X_test)
        mse = np.mean((y_test - y_pred_tree)**2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((y_test - y_pred_tree)**2) / np.sum((y_test - np.mean(y_test))**2)
        results_tree.append((mse, rmse, r2))

        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        mse = np.mean((y_test - y_pred_rf)**2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((y_test - y_pred_rf)**2) / np.sum((y_test - np.mean(y_test))**2)
        results_rf.append((mse, rmse, r2))

    # Average results
    avg_linear = np.mean(results_linear, axis=0)
    avg_tree = np.mean(results_tree, axis=0)
    avg_rf = np.mean(results_rf, axis=0)

    print("Linear Regression (avg):", avg_linear)
    print("Decision Tree (avg):", avg_tree)
    print("Random Forest (avg):", avg_rf)
    return avg_linear, avg_tree, avg_rf
