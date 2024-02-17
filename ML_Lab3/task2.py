
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math
import matplotlib.pyplot as plt

def predict_using_sklearn():
    df = pd.read_csv("test_scores.csv")
    r = LinearRegression()
    r.fit(df[['math']], df.cs)
    return r.coef_[0], r.intercept_

def gradient_descent(x, y):
    m_curr = 0
    b_curr = 0
    iterations = 10
    n = len(x)
    learning_rate = 0.0002

    x_min, x_max = np.min(x), np.max(x)
    x_range = np.linspace(x_min, x_max, 100)

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([value**2 for value in (y - y_predicted)])
        md = -(2/n) * sum(x * (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        # Plotting the linear regression line during gradient descent
        plt.scatter(x, y, color='blue', label='Data points')
        plt.plot(x_range, m_curr * x_range + b_curr, color='red', label='Regression line')
        plt.xlabel('Math Scores')
        plt.ylabel('CS Scores')
        plt.title(f'Gradient Descent - Iteration {i+1}')
        plt.legend()
        plt.show()

    return m_curr, b_curr

if __name__ == "__main__":
    df = pd.read_csv("test_scores.csv")
    x = np.array(df.math)
    y = np.array(df.cs)

    m, b = gradient_descent(x, y)
    print("Using gradient descent function: Coef {} Intercept {}".format(m, b))

    m_sklearn, b_sklearn = predict_using_sklearn()
    print("Using sklearn: Coef {} Intercept {}".format(m_sklearn, b_sklearn))