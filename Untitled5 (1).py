#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate some sample data
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(80)

# Fit a linear regression model to the data
linear_reg = LinearRegression()
linear_reg.fit(X, y)

# Generate a range of X values for prediction
X_range = np.arange(0, 5, 0.1)[:, np.newaxis]

# Transform the input data to include polynomial features (e.g., X^2)
degree = 3  # You can change the degree to control the polynomial order
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

# Fit a polynomial regression model to the transformed data
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# Predict the values for the X_range using the polynomial regression model
X_range_poly = poly_features.transform(X_range)
y_poly_pred = poly_reg.predict(X_range_poly)

# Plot the data and regression lines
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_range, linear_reg.predict(X_range), color='green', label='Linear Regression')
plt.plot(X_range, y_poly_pred, color='red', label='Polynomial Regression (Degree {})'.format(degree))
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()


# In[ ]:




