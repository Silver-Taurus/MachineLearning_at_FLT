''' Decision Tree Regression '''

# ***** Decision-Tree-Regression Intution *****
# (CART: Classification and Regression Trees)
#   - We have two types of Tree algos in ML:
#       - Classification Trees
#       - Regression Trees
#
#   - Here, we are going to use the Regression Trees:
#       - Here, the information gets splitted on the basis of algo `information entropy` and stops dividing when 
#         there is no more information left to be get splitted and hence we get the optimal splits and the terminal
#         leaves. This is the concept of `Tree`.
#       - Now, how the `Decision` works is: Suppose we have 4 optimal splits for an information having two features
#         x and y and a dependent variable (in a 2D - scatter plot), say:
#           - 1st split: x = 20
#           - 2nd split: y = 170
#           - 3rd split: y = 200
#           - 4th split: x = 40
#         So, now we got out terminal leaves. Now, whenever we got a value (say, X(x0, y0)), we will check:
#                                   x0 < 20
#                                      /\
#                                (Yes)/  \(No)
#                                    /    \
#                                   /      \
#                              y0 < 200    y0 < 170
#                              /\               /\
#                        (Yes)/  \(No)         /  \
#                            /    \      (Yes)/    \(No)
#                           /      \         /      \
#                        [T-1]    [T-2]  x0 < 40   [T-5]
#                                          /\
#                                         /  \
#                                        /    \
#                                       /      \
#                                     [T-3]   [T-4]
#
#   - So from the above representation of the Decison-Tree, we get the five Terminal leaves which helps in
#     predicting the output. But each of the terminal leaf above is a whole region, so how we are going to get
#     the continous-type value is the main question - This happens as, whenever a point falls in a terminal leaf 
#     region, then the average values of all the points present in that region is assigned to that point.
#
# This is ofcourse a Non-Linear Regression.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Making the feature matrix and target vector
features = dataset.iloc[:, [1]].values
target = dataset.iloc[:, -1].values

# Fitting the Decision Tree Regressor to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(features, target)

# Fitting the Decision Tree Regression to the dataset
predicted_target = regressor.predict(np.array([[6.5]]))

# Visualising the Decision Tree Regression results
#
#   plt.scatter(features, target, color='red')
#   plt.plot(features, regressor.predict(features), color='blue')
#   plt.title('Truth or Bluff (Decision Tree Regression)')
#   plt.xlabel('Position level')
#   plt.ylabel('Salary')
#   plt.show()
#
#   -From the above curve, we got the linear lines between the two points as we are giving only the ten levels of 
#    inputs.
# 
#   - As in the case of Polynomial Regresion we got a curve but since the values to be plot are low we used grid 
#     values (i.e., Providing a lower resolution). But it is was not much necessary there, as it is a Non-Linear 
#     Continuous Regression (before this we have dealt with Linear Continuous Regressions).
#
#   - But here, we are dealing with a new type of Regression - Non-Linear Non-Continuous Regression. As from the
#     intitution of the Decision Tree Regression algo we got to know that for each level we got an average value
#     so unless we provide a `lower resolution` (since in this case we have less information) we are not gonna get
#     a real non-continuous decision-tree curve.
#
# The Real Decision Tree Regression results
feature_grid = np.arange(min(features), max(features), 0.01)
feature_grid = feature_grid.reshape((len(feature_grid), 1))
plt.scatter(features, target, color='red')
plt.plot(feature_grid, regressor.predict(feature_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
  