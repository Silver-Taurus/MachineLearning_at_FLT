''' Random Forest Regression '''

# ***** Random-Forest-Regression Intution *****
# (Ensemble Learning Algorithms: It is type of learning in which you put the same algorithm multiple times or combine
#  different algorithms to make something much more powerful than the original one.)
#   - Step1: Pick at random `k` data points from the Training set.
#   - Step2: Build the Decision Tree associated to these `k` data points.
#   - Step3: Choose the number `n` (number of trees you want to build) and repeat steps1 and 2.
#   - Step4: For a new data point, make each one of your `n` trees predict the value of `y` and assign the new data
#            point the average value of all the predicted `y` values.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Making the features matrix and target vector
features = dataset.iloc[: , [1]].values
target = dataset.iloc[:, -1].values

# Fitting the Random Forest Regressor to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(features, target)

# Fitting the Decision Tree Regression to the dataset
predicted_target = regressor.predict(np.array([[6.5]]))

# Visualising the Random Forest Regressor results
#   - Since, it is also a `non-continuous` non-linear regression we need to provide lower resolution to see the real
#     results. 
feature_grid = np.arange(min(features), max(features), 0.001)
feature_grid = feature_grid.reshape((len(feature_grid), 1))
plt.scatter(features, target, color='red')
plt.plot(feature_grid, regressor.predict(feature_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
  