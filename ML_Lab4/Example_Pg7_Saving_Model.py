import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
df=pd.read_csv('homeprices.csv')
print(df )

#matplotlib inline
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')
plt.legend()
plt.show()

new_df=df.drop('price',axis='columns')
print(new_df)
price=df.price

reg=linear_model.LinearRegression()
reg.fit(new_df,price)

import pickle
with open ('model_pickle','wb') as file:
    pickle.dump(reg,file)
with open('model_pickle','rb') as file:
    mp=pickle.load(file)
print(mp.coef_)
print(mp.intercept_)
print(mp.predict([[5000]]))


