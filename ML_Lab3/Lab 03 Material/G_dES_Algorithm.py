import pandas as pd
import numpy as np
import random

def initialize(dim):
    b = random.random()
    theta = np.random.rand(dim)
    return b, theta

b, theta = initialize(3)
print("Bias:", b, "Weights:", theta)

import math

def predict_Y(b, theta, X):
    return b + np.dot(X, theta)

Y_hat = predict_Y(b, theta, 5)
print(Y_hat[0:10])

import math

def get_cost(Y, Y_hat):
    Y_resd = Y - Y_hat
    return np.sum(np.dot(Y_resd.T, Y_resd)) / len(Y - Y_resd)

Y_hat = predict_Y(b, theta, 5)
get_cost(Y, Y_hat)

def update_theta(x, y, y_hat, b_0, theta_o, learning_rate):
    db = (np.sum(y_hat - y) * 2) / len(y)
    dw = (np.dot((y_hat - y), x) * 2) / len(y)
    b_1 = b_0 - learning_rate * db
    theta_1 = theta_o - learning_rate * dw
    return b_1, theta_1

print("After initialization -Bias: ", b, "theta: ", theta)

Y_hat = predict_Y(b, theta, X)
b, theta = update_theta(X, Y, Y_hat, b, theta, 0.01)

print("After first update -Bias: ", b, "theta: ", theta)

get_cost(Y, Y_hat)

