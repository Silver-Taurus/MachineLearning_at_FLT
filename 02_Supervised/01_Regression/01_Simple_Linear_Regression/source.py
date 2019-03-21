''' Simple Linear Regression '''

# ***** Simple-Linear-Regression Intution *****
# Here, we have a simple formula: `y = b0 + b1.x`
# where, x = independent variable
#        y = dependent variable
#        b1 = quantifier of how `y` will change with the unit change in `x`.        
#        b0 = the constant quantifier for the case where rest of the expression evaluates to zero but 
#             still we are left with some quantifier.
#
# Now, Since the from the above algo we got a predicted value for a given data, we need to converge our
# predictions to the actual target values. So, we can converge the line to give more accurate results using
# one of the most commly used techniques of error calculation --> Ordinary Least Square (OLS) error 
# calculation technique. Mathematically represented as: `SSres = SUM((y-y^)**2)` where, 
# (SSres  -->  Sum of Squares of residuals)
 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Salary_Data.csv')

# Making feature matrix and target vector
features = dataset.iloc[:, :-1].values       # This creates a matrix
target = dataset.iloc[:, -1]                  # This creates a vector

# Splitting the dataset
from sklearn.model_selection import train_test_split
training_features, testing_features, training_target, testing_target = train_test_split(features, target, 
                                                                                        test_size=0.2,
                                                                                        random_state=0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(training_features, training_target)

# Predicting the Test set results
predicted_target = regressor.predict(testing_features)

# Calculating the Error value (just for reference)
error = abs(testing_target - predicted_target)

# ***** Visualising Results *****
#   - Visualising the Training set results
plt.subplot(121)
plt.scatter(training_features, training_target, color='red')
plt.plot(training_features, regressor.predict(training_features), color='blue')
plt.title('Profit vs R&D-Spend (Training-Set)')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
#   - Visualising the Test set results
plt.subplot(122)
plt.scatter(testing_features, testing_target, color='red')
plt.plot(testing_features, predicted_target, color='blue')
plt.title('Profit vs R&D-Spend (Testing-Set)')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.show()



# ***** R-Squared Intution *****
# As, we generally uses the OLS method for error calculation, but we got more error calculation methods, out of which the one
# is `R-Squared`.
#
#   - As we know, in the case of OLS, we have: `SSres = SUM((y-y^)**2)`
#
#   - Now for R-Squared firstly we have SSres and secondly we have: `SStot = SUM((y-y_avg)**2)` where,
#     (SStot  --> Total Sum of Squares) and then the `R-Squared` is: ```R**2 = 1 - (SSres / SStot)``` 
#
#     So R-sqaured tell you how good your modeled line is in comparison to your `average line`. So for the modeled line to be
#     a good line your SSres should be low, i.e., the more points should fall on or near to the modeled line. The more lesser
#     the value of SSres the more lower the value will be lower for `SSres / SStot` and hence the larger the value will be for
#     R. In the ideal case when all the points lie on the line (that never happens in real scenario) SSres = 0 which makes 
#     R = 1. So for this we got to know:
#       - The larger the R or R**2 value is the more accurate the modeled line is.
#
# Also the R**2 can be negative in the case when your modeled line is giving the results worse than your average line.
# This can be possible in the case, when your line is extending in a diiferent direction than that of the average line and
# this shows that your model is totally broken and useless.
#
# The same concept of R-Squared applies on other Linear Regression Models as well.

# ***** Adjusted R-Squared Intution *****
# Now as we know that the R**2 is said as quantifier of `Goodness of fit`, the greater it is the better is your model.
#
# But here comes a problem:
#   - When there are more feature variables, SSres always decreases and thus R**2 will never decrease, either it will remain 
#     same or increase. 
#   - But the problem is that, even though the extra addded variable is not related to the dependent variable, then also
#     there will still be some random coefficient that will be gonna provided to that variable. So because of that random 
#     coefficient the R**2 will gonna take some random co-relation and thus the value of R**2 may increase but will never
#     decrease.
#
# So Your R-squared is biased on the number of features you are providing.
#
# Here comes the `Adjusted-R-Squared`:  ``` Adj R**2 = 1 - (1 - R**2).(n-1 / n-p-1) ```
# where, n = sample size
#        p = number of regressors (variables)
#
# In this case, if R**2 is large then `1 - R**2` will be less which is then mulitplied by `penalizing factor` and then 
# subtracted from 1 making it large again. So for Adjusted R**2 to be large (for a good model) we need large value of R**2
# and less value of penalizing factor.
#
# If your variable does not helping much then the R**2 terms increases less while the `p` will make the ratio larger resulting
# in the larger term after multiplication and hence after subtraction from 1, the Adjusted R**2 value will decrease.
#
# It is a good metric which will tell us that the variables that we are choosing to build the model for predicting the target
# values are needed or not.
