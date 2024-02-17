# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# temperature = np.random.rand(100, 1) * 30
# ice_cream_sales = 50 - (temperature - 20)**2 + np.random.randn(100, 1) * 5
# plt.scatter(temperature, ice_cream_sales)
# plt.xlabel('Temperature (°C)')
# plt.ylabel('Ice Cream Sales')
# plt.show()
# # Apply polynomial regression with different degrees
# degrees = [1, 2, 3, 4, 5]
# for degree in degrees:
#     # Transform features into polynomial features
#     poly_features = PolynomialFeatures(degree=degree, include_bias=False)
#     X_poly = poly_features.fit_transform(temperature)
#
#     # Fit linear regression model
#     model = LinearRegression()
#     model.fit(X_poly, ice_cream_sales)
#
#     # Visualize the results
#     plt.scatter(temperature, ice_cream_sales, label='Original Data')
#     plt.xlabel('Temperature (°C)')
#     plt.ylabel('Ice Cream Sales')
#
#     # Create a range of temperatures for predictions
#     temperature_range = np.linspace(temperature.min(), temperature.max(), 100).reshape(-1, 1)
#     X_poly_range = poly_features.transform(temperature_range)
#
#     # Make predictions
#     ice_cream_sales_pred = model.predict(X_poly_range)
#
#     # Plot the regression line
#     plt.plot(temperature_range, ice_cream_sales_pred, label=f'Degree {degree}')
#
#     plt.legend()
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic dataset
np.random.seed(42)
temperature = np.random.rand(100, 1) * 30
ice_cream_sales = 50 - (temperature - 20) ** 2 + np.random.randn(100, 1) * 5

# Plot the original data points
plt.scatter(temperature, ice_cream_sales)
plt.xlabel('Temperature (°C)')
plt.ylabel('Ice Cream Sales')

# Try different polynomial degrees and visualize results
for degree in [1, 2, 3, 4, 5]:
    # Apply polynomial regression
    polynomial_features = PolynomialFeatures(degree=degree)
    X_poly = polynomial_features.fit_transform(temperature)

    model = LinearRegression()
    model.fit(X_poly, ice_cream_sales)

    # Make predictions
    predictions = model.predict(X_poly)

    # Calculate and print the mean squared error
    mse = mean_squared_error(ice_cream_sales, predictions)
    print(f'Degree {degree} - Mean Squared Error: {mse}')

    # Visualize the results
    plt.plot(temperature, predictions, label=f'Degree {degree}')

# Display the legend
plt.legend()

# Show the plot
plt.show()