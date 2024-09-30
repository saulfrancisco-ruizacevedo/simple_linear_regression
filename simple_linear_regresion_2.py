import numpy as np

def normalize_data(x: np.ndarray) -> tuple:
    """Normalize the input data and return the mean and standard deviation."""
    mu = np.mean(x)
    sigma = np.std(x)
    x_norm = (x - mu) / sigma
    return x_norm, mu, sigma

def compute_gradients(x: np.ndarray, y: np.ndarray, w: float, b: float) -> tuple:
    """Calculate the gradients for w and b."""
    y_hat = w * x + b
    error = y_hat - y
    dj_dw = np.mean(error * x)   # Gradient with respect to w
    dj_db = np.mean(error)       # Gradient with respect to b
    return dj_dw, dj_db

def gradient_descent(x: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple:
    """Perform gradient descent to optimize w and b."""
    w = 0.0  # Slope
    b = 0.0  # Intercept

    for epoch in range(iterations):
        dj_dw, dj_db = compute_gradients(x, y, w, b)

        # Update parameters
        w -= learning_rate * dj_dw
        b -= learning_rate * dj_db

        if epoch % (iterations // 10) == 0:
            print(f"{epoch} w: {w} b: {b}")

    return w, b

def rescale_parameters(w: float, b: float, sigma: float, mu: float) -> tuple:
    """Descale w and adjust b."""
    w_rescaled = w / sigma
    b_rescaled = b - (w_rescaled * mu)
    return w_rescaled, b_rescaled

# Input data
x = np.array([173, 171, 189, 181])
y = np.array([81, 72, 96, 94])

# Normalization
x_norm, mu, sigma = normalize_data(x)

# Learning parameters
learning_rate = 1.0e-3
iterations = 100000

# Perform gradient descent
w_final, b_final = gradient_descent(x_norm, y, learning_rate, iterations)

# Rescale w and adjust b
w_rescaled, b_rescaled = rescale_parameters(w_final, b_final, sigma, mu)

print(f"Final w (rescaled): {w_rescaled}, Final b (rescaled): {b_rescaled}")
