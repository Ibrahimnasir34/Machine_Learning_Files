import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
hours_studied = np.random.rand(100, 1) * 10
exam_scores = 2 * hours_studied + 5 + np.random.randn(100, 1) * 2
plt.scatter(hours_studied, exam_scores)
plt.xlabel('Hours Studied')
plt.ylabel('Exam Scores')
plt.show()
print(hours_studied)
print(exam_scores)