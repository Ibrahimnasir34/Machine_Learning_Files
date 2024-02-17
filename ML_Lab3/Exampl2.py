import numpy as np
def stochastic_gradient_descent(X, y, learning_rate, num_epochs):
    theta = np.zeros(X.shape[1])
    m = len(y)

    for epoch in range(num_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index + 1]
            yi = y[random_index:random_index + 1]

            prediction = np.dot(xi, theta)
            error = prediction - yi
            gradient = xi.T.dot(error)
            theta -= learning_rate * gradient / m

    predictions = X.dot(theta)
    errors = predictions - y
    cost = np.sum(errors ** 2) / (2 * m)
    print(f"Epoch {epoch + 1}/{num_epochs}, Cost: {cost}")
    return theta


# Example usage for stochastic gradient descent
X = np.random.rand(100, 2)
y = 2 * X[:, 0] + 3 * X[:, 1] + 1 + 0.1 * np.random.randn(100)
X_b = np.c_[np.ones((100, 1)), X]
learning_rate = 0.01
num_epochs = 50
theta = stochastic_gradient_descent(X_b, y, learning_rate, num_epochs)
print("Final Coefficients:", theta)