''' Polynomial Linear Regression '''

# ***** Polynomial-Linear-Regression Intution *****
# Here, we have a simple formula: `y = b0 + b1.x1 + b2.x1^2 + ... + bn.x1^n`
# where, x = independent variable
#        y = dependent variable
#        b1, b2, ... = quantifier of how `y` will change with the unit change in `x`.        
#        b0 = the constant quantifier for the case where rest of the expression evaluates to zero but still we are left with
#             some quantifier.
#
# This is used in the case where we have a curve-fitted graph.
#
# Why it is still called Linear?
#   - Here, when we are talking about Linear, we are not denoting the x (since x is polynomial) but what we are denoting 
#     here is the class of regression that is the coefficients that are being used with the independent variables.
#     So, we can replace the coefficients with other coefficients to turn the equation into a linear one.
#
# But the way it fits the data is Non-Linear in nature.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Making feature matrix and target vector
features = dataset.iloc[:, [1]].values
target = dataset.iloc[:, -1].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(features, target)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree=5)     # By default also, the degree is 2.
poly_features = poly_regressor.fit_transform(features)
poly_lin_regressor = LinearRegression()
poly_lin_regressor.fit(poly_features, target)

# *****Visualise the dataset *****
plt.subplot(121)
feature_grid = np.arange(min(features), max(features), 0.1)
feature_grid = feature_grid.reshape((len(feature_grid), 1))
plt.scatter(features, target, color='red')
plt.plot(feature_grid, linear_regressor.predict(feature_grid), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.subplot(122)
plt.scatter(features, target, color='red')
plt.plot(feature_grid, poly_lin_regressor.predict(poly_regressor.fit_transform(feature_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Linear Regression)')
plt.show()
