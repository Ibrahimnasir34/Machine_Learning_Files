import matplotlib.pyplot as plt
from scipy import stats
x=[5,7,8,7,2,17,2,9,4,11,12,9,6]
y=[99,86,87,88,11,86,103,87,94,78,77,85,86]
slope,intercept,r,p,std_err=stats.linregress(x,y)
#relation r intercept  c ,slope m  lingress compute the  m,c

def myfunc(x):
    return slope *x +intercept
# each value of x through y every new value

speed=myfunc(10)
print(speed)



