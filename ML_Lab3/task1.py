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
import matplotlib.pyplot as plt

def gradient_descent(x, y):
    m_curr = b_curr = 0
    rate = 0.01
    n = len(x)
    plt.scatter(x, y, color='red', marker='+', linewidth=5)  # Fix the linewidth value

    for i in range(100):
        y_predicted = m_curr * x + b_curr
        plt.plot(x, y_predicted, color='green')

        md = -(2/n) * sum(x * (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)

        m_curr = m_curr - rate * md
        b_curr = b_curr - rate * bd

    plt.show()

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])
gradient_descent(x, y)