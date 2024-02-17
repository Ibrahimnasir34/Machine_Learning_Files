import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

df = pd.read_csv('Advertising.csv')
print(df.head())

# Normalize the x, y
X = df[['TV', 'radio', 'newspaper']]
Y = df['sales']  # Using 'sales' as the correct column name
Y = np.array((Y - Y.mean()) / Y.std())
X = X.apply(lambda rec: (rec - rec.mean()) / rec.std(), axis=0)
