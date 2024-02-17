import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
hours_studied = np.random.rand(100, 1) * 10
exam_scores = 2 * hours_studied + 5 + np.random.randn(100, 1) * 2
plt.scatter(hours_studied, exam_scores)
plt.xlabel('Hours Studied')
plt.ylabel('Exam Scores')
plt.show()
from sklearn.linear_model import LinearRegression


model = LinearRegression()


model.fit(hours_studied, exam_scores)


slope = model.coef_[0]
intercept = model.intercept_

print(f"Slope: {slope[0]}, Intercept: {intercept}")
# Predict exam scores using the model
predicted_scores = model.predict(hours_studied)

# Plot the original data points
plt.scatter(hours_studied, exam_scores, label='Original Data')

# Plot the regression line
plt.plot(hours_studied, predicted_scores, color='red', label='Regression Line')

plt.xlabel('Hours Studied')
plt.ylabel('Exam Scores')
plt.legend()
plt.show()
