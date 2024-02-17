#from sklearn.externals import joblib
import pickle
# with open ('model_pickle','wb') as file:
#     pickle.dump(reg,file)
with open('model_pickle','rb') as file:
    mp=pickle.load(file)
print(mp.coef_)
print(mp.intercept_)
print(mp.predict([[5000]]))
