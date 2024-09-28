# Housing Prices Prediction using Linear Regression

This project demonstrates how to predict housing prices based on the size of the house using a technique called **linear regression**. The code is written in Python 3 and uses libraries like NumPy and Matplotlib to perform calculations and visualize the results.

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
python housing_prices.py
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

