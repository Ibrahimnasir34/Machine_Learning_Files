import numpy as np

def gradient_descent(X, y, learning_rate, num_iterations):
    # Initialize coefficients
    theta = np.zeros(X.shape[1])
    m = len(y)

    for iteration in range(num_iterations):
        # Calculate predictions
        predictions = np.dot(X, theta)

        # Calculate errors
        errors = predictions - y

        # Update coefficients
        gradient = np.dot(X.T, errors) / m
        theta -= learning_rate * gradient

        # Calculate and print the cost
        cost = np.sum(errors ** 2) / (2 * m)
        print(f"Iteration {iteration + 1}/{num_iterations}, Cost: {cost}")

    return theta

# Usage example
X = np.random.rand(100, 1)  # Example feature
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)  # Example linear relationship with noise
# Add a column of ones to X for the intercept term
X_b = np.c_[np.ones((100, 1)), X]
# Transpose y to make it a column vector
y = y.T[0]
# Apply gradient descent
learning_rate = 0.01
num_iterations = 100
theta = gradient_descent(X_b, y, learning_rate, num_iterations)
print("Final Coefficients:", theta)

# Calculate and print the final cost using the trained model
final_predictions = np.dot(X_b, theta)
final_cost = np.sum((final_predictions - y) ** 2) / (2 * len(y))
print("Final Cost:", final_cost)
