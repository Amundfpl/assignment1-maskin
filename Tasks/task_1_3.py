import numpy as np
import matplotlib.pyplot as plt

def run_task_1_3(df):
    """
    Fit a simple linear regression model using gradient descent
    to predict wine quality from chlorides.
    """
    # Extract variables unnomralized
    #X = df["chlorides"].values
    #y = df["quality"].values

    # Standardize X
    X = (df["chlorides"].values - df["chlorides"].mean()) / df["chlorides"].std()
    y = df["quality"].values

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

    print("\nFinal model: quality = {:.4f} * chlorides + {:.4f}".format(w, b))

    # Plot results
    plt.scatter(X, y, label="Data", alpha=0.5)
    plt.plot(X, w*X + b, color="red", label="Fitted line")
    plt.xlabel("Chlorides")
    plt.ylabel("Quality")
    plt.title("Task 1.3: Linear Regression (Gradient Descent)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Images/task_1_3_regression.png", dpi=300)
    plt.show()
