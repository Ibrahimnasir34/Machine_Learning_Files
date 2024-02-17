# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt
# # Load the dataset
# # Assuming the dataset is stored in a CSV file named 'student_data.csv'
# dataset = pd.read_csv('student_data.csv')
# # Split the dataset into features (X) and target variable (y)
# X = dataset[['hours_of_study', 'attendance', 'extracurricular_activities']]
# y = dataset['grades']
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # Fit a linear regression model to the training data
# model = LinearRegression()
# model.fit(X_train, y_train)
# # Predict the grades on the testing data
# y_pred = model.predict(X_test)
# # Evaluate the model's performance using mean squared error
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')
# # Plot actual vs. predicted grades
# plt.scatter(y_test, y_pred)
# plt.xlabel('Actual Grades')
# plt.ylabel('Predicted Grades')
# plt.title('Actual vs. Predicted Grades')
# plt.show()
# # Plot the coefficients of the model
# coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
# coefficients.plot(x='Feature', y='Coefficient', kind='bar')
# plt.xlabel('Feature')
# plt.ylabel('Coefficient')
# plt.title('Model Coefficients')
# plt.show()
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create a DataFrame from the provided data
data = {
    "Hours of Study": [5, 8, 4, 7, 6, 10, 3, 9, 5, 8],
    "Attendance (%)": [85, 92, 80, 88, 90, 95, 75, 94, 82, 89],
    "Extracurricular Activities": [2, 3, 1, 2, 1, 3, 1, 2, 1, 3],
    "Grades": [75, 90, 60, 80, 70, 95, 50, 85, 65, 88]
}

df = pd.DataFrame(data)

# Define X (features) and y (target variable)
X = df[["Hours of Study", "Attendance (%)", "Extracurricular Activities"]]
y = df["Grades"]

# Create and train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions on unseen data (assuming you have new data)
# new_data = ...  # Replace with your new data
# predictions = model.predict(new_data)

# Evaluate the model's performance on the training data
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Mean squared error:", mse)
print("R-squared score:", r2)

# Print the model coefficients (weights for each feature)
print("Model coefficients:", model.coef_)