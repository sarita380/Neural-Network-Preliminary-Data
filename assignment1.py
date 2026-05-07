import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import os

# =========================================================
# FILE PATH ( weather_data.csv )
# =========================================================
datafile = "weather_data.csv"

# =========================================================
# Problem 1 (30 pts): Normal Equation
# =========================================================
def normal_equation_line(x, y):
    """
    Fit y = theta0 + theta1*x using Normal Equation.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

    X = np.hstack([np.ones((x.shape[0], 1)), x])  # add bias column
    theta = np.linalg.inv(X.T @ X) @ (X.T @ y)

    theta0 = float(theta[0, 0])
    theta1 = float(theta[1, 0])
    return theta0, theta1


def predict_line(theta0, theta1, x_values):
    x_values = np.asarray(x_values, dtype=np.float64)
    return theta0 + theta1 * x_values


def run_problem_1_detailed():
    marks = np.array([95, 85, 80, 70, 60], dtype=np.float64)
    grades = np.array([85, 95, 70, 65, 70], dtype=np.float64)

    # Step 1: Build X and y
    X = np.c_[np.ones((len(marks), 1)), marks.reshape(-1, 1)]
    y = grades.reshape(-1, 1)

    # Step 2: Compute X^T X
    XtX = X.T @ X

    # Step 3: Compute X^T y
    Xty = X.T @ y

    # Step 4: Compute theta = (X^T X)^-1 X^T y
    theta = np.linalg.inv(XtX) @ Xty
    theta0, theta1 = float(theta[0, 0]), float(theta[1, 0])

    print("\n====================")
    print("Problem 1 (Normal Equation) - Detailed Steps")
    print("====================")
    print("X =\n", X)
    print("y =\n", y)
    print("\nX^T X =\n", XtX)
    print("\nX^T y =\n", Xty)
    print("\nTheta = (X^T X)^-1 (X^T y) =\n", theta)

    print("\nFinal parameters:")
    print(f"theta0 (intercept) = {theta0:.6f}")
    print(f"theta1 (slope)     = {theta1:.6f}")
    print(f"Final line: y_hat = {theta0:.6f} + {theta1:.6f}x")

    # Predict required marks
    given_marks = [65, 75, 77, 83, 87]
    preds = predict_line(theta0, theta1, given_marks)

    print("\nPredicted grades:")
    for m, p in zip(given_marks, preds):
        print(f"Marks {m:>3} -> Predicted Grade {p:.4f}")


# =========================================================
# Problem 2: Weather Linear Regression (Raw Python)
# =========================================================

# a) get_data()
def get_data(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"ERROR: Cannot find file '{filename}'.\n"
            f"Make sure weather_data.csv is in the same folder as assignment1.py."
        )

    df = pd.read_csv(filename)

    # EXACT column names from CSV
    X_ = df[["Humidity", "Visibility (km)"]].values.astype(np.float64)
    Y_ = df[["Temperature (C)"]].values.astype(np.float64)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_, Y_, test_size=0.25, random_state=42
    )
    return X_train, X_test, y_train, y_test


# a) data_iter()
def data_iter(batch_size, X, y):
    num_examples = len(X)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        batch_idx = indices[i:i + batch_size]
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]
        yield X_batch, y_batch

# b) create_model_parameter()
def create_model_parameter(mu, sigma, row, column):
    w = np.random.normal(mu, sigma, size=(row, column))
    b = np.zeros((1, column))
    return w, b

# c) model()
def model(X, w, b):
    return X @ w + b

# d) squared_loss()
def squared_loss(y_hat, y):
    return 0.5 * np.mean((y_hat - y) ** 2)

# e) gradient()
def gradient(X, y, w, b):
    m = X.shape[0]
    y_hat = model(X, w, b)
    err = y_hat - y  # (m,1)

    grad_w = (X.T @ err) / m
    grad_b = np.sum(err, axis=0, keepdims=True) / m
    return grad_w, grad_b


# f) sgd()
def sgd(params, grads, lr):
    w, b = params
    grad_w, grad_b = grads
    w = w - lr * grad_w
    b = b - lr * grad_b
    return w, b

# Standardization
def standardize_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-12
    return mu, sigma

def standardize_transform(X, mu, sigma):
    return (X - mu) / sigma

# g) train()
def train(lr, num_epochs, X_train, y_train, batch_size, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # Standardize features
    Xmu, Xsig = standardize_fit(X_train)
    Xs = standardize_transform(X_train, Xmu, Xsig)

    # Initialize parameters
    w, b = create_model_parameter(mu=0.0, sigma=0.01, row=Xs.shape[1], column=1)

    losses = []
    for epoch in range(num_epochs):
        for Xb, yb in data_iter(batch_size, Xs, y_train):
            grads = gradient(Xb, yb, w, b)
            w, b = sgd((w, b), grads, lr)

        train_loss = squared_loss(model(Xs, w, b), y_train)
        losses.append(train_loss)
        print(f"epoch {epoch + 1:02d}, batch {batch_size:<6}, loss {train_loss:.6f}")

    return (w, b, Xmu, Xsig), losses

# Predict helper
def predict(X, params_pack):
    w, b, Xmu, Xsig = params_pack
    Xs = standardize_transform(X, Xmu, Xsig)
    return model(Xs, w, b)

# h) draw_loss() 3 batch sizes
def draw_loss(loss_dict):
    """
    loss_dict: {batch_size: losses_list}
    """
    plt.figure()
    for bs, losses in loss_dict.items():
        epochs = np.arange(1, len(losses) + 1)
        plt.plot(epochs, losses, label=f"batch_size={bs}")

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epochs (3 Batch Sizes)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def explain_batch_effect():
    print("\n====================")
    print("Effect of Batch Size on Training Loss")
    print("====================")
    print("Small batch: More updates per epoch -> can learn faster but loss curve is noisier.")
    print("Large batch: Fewer updates -> smoother curve, sometimes slower improvement per epoch.")
    print("Exact/full batch: Smoothest curve, but each step is expensive and fewer updates occur.")


def test_model(X_test, y_test, params_pack, n=5):
    y_pred = predict(X_test, params_pack)
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"\nTest MSE = {mse:.6f}")

    print("\nSample predictions:")
    for i in range(min(n, len(y_test))):
        print(f"X={X_test[i]} | y_true={y_test[i,0]:.3f} | y_pred={y_pred[i,0]:.3f}")

def run_problem_2():
    print("\n===============================")
    print("Problem 2 (Weather Linear Regression)")
    print("=================================")

    X_train, X_test, y_train, y_test = get_data(datafile)
    print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    lr = 0.1
    num_epochs = 30

    # Choose 3 batch sizes: small, large, exact
    bs_small = 32
    bs_large = 2048
    bs_exact = X_train.shape[0]  # exact/full batch

    print("\n--- Training SMALL batch ---")
    params_small, losses_small = train(lr, num_epochs, X_train, y_train, bs_small)

    print("\n--- Training LARGE batch ---")
    params_large, losses_large = train(lr, num_epochs, X_train, y_train, bs_large)

    print("\n--- Training EXACT (full) batch ---")
    params_exact, losses_exact = train(lr, num_epochs, X_train, y_train, bs_exact)

    # Single figure with all 3 batch
    draw_loss({
        bs_small: losses_small,
        bs_large: losses_large,
        bs_exact: losses_exact
    })

    explain_batch_effect()

    # Evaluate using the small batch model
    test_model(X_test, y_test, params_small)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    run_problem_1_detailed()
    run_problem_2()
