import numpy
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
numpy.random.seed(2)
x=numpy.random.normal(3,1,100)
y=numpy.random.normal(150,40,100)/x
train_x=x[:80]
train_y=y[:80]
test_x=x[80:]
test_y=y[80:]
mymodel=numpy.poly1d(numpy.polyfit(train_x,train_y,4))
r2=r2_score(train_y,mymodel(train_x))
print(r2)
predicted_y=mymodel(train_x)
plt.scatter(train_x,train_y,label='Acutcaol')
plt.plot(train_x,predicted_y,label='Predicted',color='Red')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


