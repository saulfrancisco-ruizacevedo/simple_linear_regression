import numpy as np
import matplotlib.pyplot as plt

def compute_cost(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    """Calculate the mean squared error for linear regression."""
    m = len(y)  # Number of training examples
    predictions = w * x + b
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)

    return cost

def compute_gradient(x: np.ndarray, y: np.ndarray, w: float, b: float) -> tuple:
    """Compute gradients for w and b."""
    m = len(y)
    predictions = w * x + b
    error = predictions - y
    dj_dw = (1 / m) * np.dot(error, x)  # Gradient w.r.t w
    dj_db = (1 / m) * np.sum(error)      # Gradient w.r.t b

    return dj_dw, dj_db

def gradient_descent(x: np.ndarray, y: np.ndarray, w_init: float, b_init: float, 
                     alpha: float, num_iters: int) -> tuple:
    """Perform gradient descent to optimize w and b."""
    w, b = w_init, b_init
    J_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        J_history.append(compute_cost(x, y, w, b))

        if i % (num_iters // 10) == 0:
            print(f"Iteration {i}: Cost {J_history[-1]:.2e}, w: {w:.3f}, b: {b:.3f}")

    return w, b, J_history

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate performance metrics."""
    m = len(y_true)
    
    # MAE
    mae = np.mean(np.abs(y_pred - y_true))
    
    # MSE
    mse = np.mean((y_pred - y_true) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }

# Training data
x_train = np.array([1, 2])  # Size in 1000 sqft
y_train = np.array([300, 500])  # Price in 1000s USD

# Regression parameters
w_init, b_init = 0.0, 0.0
iterations = 100000
alpha = 0.01  # Learning rate

# Perform gradient descent
w_final, b_final, J_hist = gradient_descent(x_train, y_train, w_init, b_init, alpha, iterations)
print(f"(w, b) found by gradient descent: ({w_final:.4f}, {b_final:.4f})")


y_pred = w_final * x_train + b_final

metrics = compute_metrics(y_train, y_pred)
print("Performance Metrics:")
print(f"MAE: {metrics['MAE']:.10f}")
print(f"MSE: {metrics['MSE']:.10f}")
print(f"RMSE: {metrics['RMSE']:.10f}")

# Plot the training data points
plt.scatter(x_train, y_train, color="red", marker="x", label="Training data", s=100)

# Plot the regression line
x_values = np.linspace(np.min(x_train) - 1, np.max(x_train) + 1, num=100)
y_values = w_final * x_values + b_final
plt.plot(x_values, y_values, label="Prediction line", linewidth=3, color="blue")

# Title and labels
plt.title("Housing Prices")
plt.ylabel("Price (in 1000s of dollars)")
plt.xlabel("Size (1000 sqft)")
plt.legend()
plt.grid()
plt.show()
