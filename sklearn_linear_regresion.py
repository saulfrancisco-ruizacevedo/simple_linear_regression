import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


X_train = np.array([173, 171, 189, 181]).reshape(-1, 1)  # Size in 1000 sqft
Y_train = np.array([81, 72, 96, 94])  # Price in 1000s USD

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

sgdr = SGDRegressor(max_iter=10000)
sgdr.fit(X_norm, Y_train)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters: \nw: {w_norm}, b:{b_norm}")


plt.scatter(X_train, Y_train, color='blue', label='Data Points')

X_line = np.linspace(min(X_train), max(X_train), 100).reshape(-1, 1)  
X_line_norm = scaler.transform(X_line) 
Y_line = sgdr.predict(X_line_norm)  

plt.plot(X_line, Y_line, color='red', label='Regression Line')

plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (1000s USD)')
plt.title('Linear Regression: Size vs Price')
plt.legend()
plt.grid(True)
plt.show()