
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
dataset_path = 'canada_per_capita_income.csv'
df = pd.read_csv(dataset_path)

# Print the first few rows of the dataset
print(df.head())

# Assume 'capita' is the target variable
# Features (X)
X = df[['year']]  # Use double square brackets to create a DataFrame

# Target variable (y)
y = df['capita']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions for the year 2020
year_2020 = [[2020]]  # Use double square brackets to create a 2D array
predicted_income_2020 = model.predict(year_2020)

print(f"Predicted per capita income for the year 2020: ${predicted_income_2020[0]:,.2f}")

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Visualize the predicted vs. actual values
plt.scatter(y_test, predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()
