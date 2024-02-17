import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
df=pd.read_csv("insurance_data.csv")
print(df.head())

#plt.scatter(df.age,df.bought_insurance,marker='+',color='red)

x_train,x_test,y_train,y_test=train_test_split(df[['age']],df.bought_insurance,train_size=0.8)
print(x_test)

model=LogisticRegression()
model.fit(x_train,y_train)

y_predictt=model.predict(x_test)
y_probability=model.predict_proba(x_test)
score=model.score(x_test,y_test)

print('Predicted Value',y_predictt)
print('Probabilty',y_probability)
print('score',score)
print('coefficent',model.coef_)
print('intercept',model.intercept_)

