# Housing Prices Prediction using Linear Regression

This project demonstrates how to predict housing prices based on the size of the house using a technique called **linear regression**. The code is written in Python 3 and uses libraries like NumPy and Matplotlib to perform calculations and visualize the results.

This repository contains two Python scripts implementing simple linear regression using gradient descent:

### 1\. `simple_linear_regression.py`

This script includes a comprehensive implementation of simple linear regression. Key features include:

-   **Cost Computation**: It contains a function called `compute_cost`, which calculates and stores the cost history of the model during training. This function helps track how the cost evolves over iterations, providing insights into the convergence of the gradient descent algorithm.
-   **Performance Metrics**: The script calculates various performance metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). These metrics help evaluate the effectiveness of the regression model.
-   **Data Visualization**: Using `matplotlib`, the script visualizes the training data points and the regression line. This graphical representation allows for a better understanding of how well the model fits the data.

### 2\. `simple_linear_regression_2.py`

This script offers a simplified implementation of simple linear regression. Key aspects include:

-   **Linear Regression**: It focuses solely on performing linear regression using gradient descent without additional functionalities.
-   **Console Output**: The results, including the learned parameters (slope and intercept), are printed to the console, providing immediate feedback on the model's performance.


## Table of Contents
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [How to Run the Code](#how-to-run-the-code)
- [Explanation of the Code](#explanation-of-the-code)

## How It Works

1. **Understanding Linear Regression**: We want to find a line that best fits our data points. This line will help us predict the price of a house based on its size. 
2. **Gradient Descent**: This is a method used to find the best values for our line (specifically the slope `w` and the intercept `b`).
3. **Calculating Error**: We check how far our predictions are from the actual prices. We want to minimize this error.
4. **Visualizing Results**: Finally, we plot our data and the regression line to see how well our model fits the data.

## Requirements

Make sure you have Python 3 installed along with the following libraries:
- NumPy
- Matplotlib

You can install them using pip:
```bash 
pip install numpy matplotlib
```

## How to Run the Code

```bash 
python simple_linear_regresion.py
```

or

```bash 
python simple_linear_regresion_2.py
```

## Explanation of the Code

### Imports

-   **NumPy**: This library helps with numerical calculations, especially with arrays.
-   **Matplotlib**: This library is used for creating visual plots.

### Functions

-   **compute\_cost**: This function calculates the mean squared error between predicted and actual housing prices.
-   **compute\_gradient**: This function computes how much we should change `w` and `b` to minimize the error.
-   **gradient\_descent**: This function iteratively updates `w` and `b` to find their optimal values using the gradient descent method.
-   **compute\_metrics**: This function calculates performance metrics (MAE, MSE, RMSE) to evaluate how well our model performs.

### Training Data

This project uses a set of training data representing housing sizes and their corresponding prices.

### Visualization

Finally, the code plots the original data points and the predicted regression line, providing a visual understanding of the model's performance.

## Why We Normalize Data

In machine learning, we normalize data to ensure that all features contribute equally to the model. Normalization rescales the input features to have a mean of 0 and a standard deviation of 1. This helps gradient descent converge faster and prevents certain features (e.g., housing sizes) from dominating the optimization process simply because they have larger numerical values.

In this project, the house sizes (`x_train`) are normalized before applying the gradient descent algorithm. Normalization is done by subtracting the mean (`mu`) and dividing by the standard deviation (`sigma`), so the data falls within a similar range. This makes the training more stable and efficient.

### Rescaling the Parameters

Once the model has been trained using normalized data, the learned parameters (`w` and `b`) must be rescaled back to their original scale to make meaningful predictions on real-world data. In the code, the final `w` and `b` are rescaled as follows:

-   **Rescaling `w`**: We divide the learned `w` by the standard deviation of the original data (`sigma`).
-   **Rescaling `b`**: We adjust `b` by subtracting the product of the rescaled `w` and the mean of the original data (`mu`).
