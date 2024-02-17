#



import numpy as np
from matplotlib import pyplot as plt

ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)

plt.title("Sample Visualization")
plt.show()


import numpy as np
import pandas as pd

def stochastic_gradient_descent(X, y, learning_rate, num_epochs):
    theta = np.zeros(X.shape[1])
    m = len(y)

    for epoch in range(num_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]

            prediction = np.dot(xi, theta)
            error = prediction - yi
            gradient = xi.T.dot(error)
            theta -= learning_rate * gradient / m

        predictions = X.dot(theta)
        errors = predictions - y
        cost = np.sum(errors**2) / (2 * m)
        print(f"Epoch {epoch + 1}/{num_epochs}, Cost: {cost}")

    return theta

# Read data from CSV file
df = pd.read_csv("Advertising.csv")
X = df.iloc[:, :-1].values  # Features (all columns except the last one)
y = df.iloc[:, -1].values  # Target variable (last column)

# Add a column of ones to X for the intercept term
X_b = np.c_[np.ones((len(y), 1)), X]

learning_rate = 0.001
num_epochs = 10

theta = stochastic_gradient_descent(X_b, y, learning_rate, num_epochs)
print("Final Coefficients:", theta)

import numpy as np

def stochastic_gradient_descent(X, y, learning_rate, num_iterations):
    theta = np.zeros(X.shape[1])
    m = len(y)
    for iteration in range(num_iterations):
        random_index = np.random.randint(m)
        xi = X[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        # prediction and error for the randomly chosen instance
        prediction = np.dot(xi, theta)
        error = prediction - yi
        # Update coefficients based on the single randomly chosen instance
        gradient = xi.T.dot(error)
        theta = theta - learning_rate * gradient.flatten()
        # Calculate and print the cost
        cost = np.sum(error**2) / 2
        print(f"Iteration {iteration + 1}/{num_iterations}, Cost: {cost}")
    return theta

# Usage example
df = pd.read_csv("Advertising.csv")
X = df.iloc[:, :-1].values  # Features (all columns except the last one)
y = df.iloc[:, -1].values
X_b = np.c_[np.ones((len(y), 1)), X]

# Apply stochastic gradient descent
learning_rate = 0.001
num_iterations = 10
theta = stochastic_gradient_descent(X_b, y, learning_rate, num_iterations)
print("Final Coefficients:", theta)

