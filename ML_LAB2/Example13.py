import pandas as pd
from sklearn import linear_model

# Read the CSV file into a DataFrame
df = pd.read_csv("data.csv")

# Define the features (X) and target variable (y)
X = df[['Weight', 'Volume']]
y = df['CO2']

# Create a linear regression model
regr = linear_model.LinearRegression()

# Train the model with the provided data
regr.fit(X, y)

# Predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3
predictedCO2 = regr.predict([[2300, 1300]])

# Print the predicted CO2 emission
print(predictedCO2)
