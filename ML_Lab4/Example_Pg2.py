import numpy as np
import matplotlib.pyplot as plt
import numpy.random

print(np.random.seed(2))
x=numpy.random.normal(3,1,100)
y=numpy.random.normal(150,40,100)/x
plt.scatter(x,y)
plt.show()
