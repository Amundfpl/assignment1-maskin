import numpy as np
import matplotlib.pyplot as plt

def run_task_1_3_2(df):
    """
    Fit a simple linear regression model using gradient descent
    to predict wine quality from alcohol.
    """
    # Extract variables
    X = df["alcohol"].values
    y = df["quality"].values

    # Normalize X (optional but helps gradient descent)
    X = (X - X.mean()) / X.std()

    # Initialize parameters
    w, b = 0.0, 0.0
    alpha = 0.01     # learning rate
    epochs = 1000
    n = len(X)

    # Gradient descent loop
    for epoch in range(epochs):
        y_pred = w * X + b
        error = y - y_pred

        dw = -(2/n) * np.dot(X, error)
        db = -(2/n) * np.sum(error)

        w -= alpha * dw
        b -= alpha * db

        if epoch % 100 == 0:
            mse = np.mean(error**2)
            print(f"Epoch {epoch}: MSE={mse:.4f}, w={w:.4f}, b={b:.4f}")

    print("\nFinal model: quality = {:.4f} * alcohol + {:.4f}".format(w, b))

    # Plot results
    plt.scatter(X, y, label="Data", alpha=0.5)
    plt.plot(X, w*X + b, color="red", label="Fitted line")
    plt.xlabel("Alcohol (standardized)")
    plt.ylabel("Quality")
    plt.title("Task 1.4: Linear Regression (Gradient Descent)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Images/task_1_4_regression.png", dpi=300)
    plt.show()
