import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("HR_comma_sep.csv")

# 1. Exploratory Data Analysis
# Identify variables with direct impact on employee retention
# For simplicity, let's assume all columns except 'left' impact retention
impactful_vars = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
impactful_vars.remove('left')

# Display the correlation matrix for exploratory analysis
correlation_matrix = df[impactful_vars + ['left']].corr()
print("Correlation Matrix:")
print(correlation_matrix)

# 2. Bar charts showing impact of employee salaries on retention
sns.countplot(x='salary', hue='left', data=df)
plt.title('Impact of Salary on Employee Retention')
plt.xlabel('Salary Level')
plt.ylabel('Number of Employees')
plt.show()

# 3. Bar charts showing correlation between department and retention
sns.countplot(x='Department', hue='left', data=df)
plt.title('Correlation between Department and Employee Retention')
plt.xlabel('Department')
plt.ylabel('Number of Employees')
plt.show()

# 4. Build Logistic Regression Model
# Select relevant columns based on exploratory analysis
selected_vars = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours',
                 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'salary']

X = df[selected_vars]
y = df['left']

# Convert categorical variables to dummy/indicator variables
X = pd.get_dummies(X, columns=['salary'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Measure the accuracy of the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy of the Logistic Regression Model:", accuracy)
