import numpy as np
import matplotlib.pyplot as plt
import numpy.random

numpy.random.seed(2)

x=numpy.random.normal(3,1,100)
y=numpy.random.normal(150,40,100)/x

train_x=x[:80]
train_y=y[:80]

test_x=x[80:]
test_y=y[80:]

mymodel=numpy.poly1d(numpy.polyfit(train_x,train_y,4))

myline=numpy.linspace(0,6,100)
plt.scatter(test_x,test_y)
plt.plot(myline,mymodel(myline))
plt.show()
# print(
# mymodel.coef)
y_pred = mymodel(test_x)
print(y_pred)

