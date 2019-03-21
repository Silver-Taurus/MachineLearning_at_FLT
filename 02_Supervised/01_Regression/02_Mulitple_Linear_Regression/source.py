''' Multiple Linear Regression '''

# ***** Multiple-Linear-Regression Intution *****
# Here, we have a simple formula: `y = b0 + b1.x1 + b2.x2 + ...`
# where, x = independent variable
#        y = dependent variable
#        b1, b2, ... = quantifier of how `y` will change with the unit change in `x`.        
#        b0 = the constant quantifier for the case where rest of the expression evaluates to zero but still we are left with
#             some quantifier.


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('50_Startups.csv')

# Making feature matrix and target vector
features = dataset.iloc[:, :-1].values       # This creates a matrix
target = dataset.iloc[:, 4]                  # This creates a vector

# ***** Dummy Variable Trap *****
#   - Since in this case, the state is a feature with the categorical data. So we are going to use the Dummy Variables for
#     `New York` and `California`. 
#
#   - Since we have only two states, we can represent the state by just one dummy variable also. This is because if the
#     state is New York we got one and if the New York is 0 then we automatically gonna know that the state is California. 
#
#   - Then for this case, our equation becomes: `y = b0 + b1.x1 + b2.x2 + b3.x3 + b4.D1`
#
#   - Now if the state is New York, we will get our New York biased output as D1 will be 1 and we have the coefficient
#     b4 for creating the bias output along with the constant b0 while if the state is California the D1 is 0 removing
#     the b4 biased value and hence we left with b0 only which will account for the bias output of California.
#     (Just like the on or off switch - b4 is the switch)
#
#   - But if we are gonna include D2 as well then the switch will be D4 + D5, that will always be in an `on` state and
#     hence we cannot differentiate whether it is for New York or California. This is kwown as Dummy Variable Trap.

# TODO: See for the new ColumnTransformer as replace for categorical_features paramter od OneHotEncoder and
#       also see the future warnings for LabelEncoder usage with just OneHotEncoder.

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
features_label_encoder = LabelEncoder()
features[:, 3] = features_label_encoder.fit_transform(features[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
features = onehotencoder.fit_transform(features).toarray()

# Avoiding the Dummy Variable Trap (though in our case the Python library can take care of it)
features = features[: , 1:]

# Splitting the dataset
from sklearn.model_selection import train_test_split
training_features, testing_features, training_target, testing_target= train_test_split(features, target, 
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

# ***** Building a Model (Step by Step) *****
#   - When we have a single variable there is no such need for building a model step by step as we have linear regression
#     do simply fit on the single feature and gives the output.
#
#   - But now, we have multiple features (columns) that are pretty much responsible for that output prediction.
#     Now first and foremost, why do we need to get rid of extra features?
#       --> Garbage in and Garbage out (i.e., if we provide too much of useless data we'll not gonna get useful output).
#       --> If our model depends on very large number of factors when not needed, accounting for them or explaining others
#           about those factors is a waste of time of both, ours and others as well as explaining such a large factors
#           itself is very difficult.  
#
#   - Methods of Building models:
#       --> All-in (generally useless and a wrong practice)
#       --> Backward Elimination
#       --> Forward Selection
#       --> Bidirectional Elimination
#       --> Score Comparison
#
# Generally Step-By-Step Regression covers the methods 2 to 4.
# Many People also refer just the Bi-Directional Elimination as a Step-By-Step Regression.
#
#   - `All-in` cases:
#       - Prior Knowledge (that these specific variables are going to be used)
#       - You have to include these varaibles and have no choice
#       - Preparing for Backward Elimination
#
#   - `Backward Elimination` steps:
#       Step1 - Select a significance level to stay in the model (eg: Sl = 0.05, i.e., 5 percent)
#       Step2 - Fit the model with all possible predictors (All-in)
#       Step3 - Consider the predictor with the highest P-value. If P > Sl, go to step 4, otherwise go to FIN
#       Step4 - Remove the predictor
#       Step5 - Fit model without this predictor (or varaible) then Repeat Step3
#       FIN - Your Model is Ready.
#
#   - `Forward Selection` steps:
#       Step1 - Select a significance level to enter the model (eg: Sl = 0.05)
#       Step2 - Fit all simple regression models y ~ x(n). Select the one with thre lowest P-value
#       Step3 - Keep this variable and fit all possible models with one extra predictor added to the one(s) you already 
#               have
#       Step4 - Consider the predictor with lowest P-value. If P < Sl, go to step3, otherwise go to FIN.
#       FIN - Keep the previous model (i.e., the model before the addition of the last insignificant variable) as the
#             final model.
#
#   - `Bi-Directional Elimination` steps:
#       Step1 - Select a significance level to enter and to stay in the model (eg: Sl_enter = 0.05, Sl_stay = 0.05)
#       Step2 - Perform the next step of Forward Selection (new variables must have: P < Sl_enter to enter)
#       Step3 - Perform All Steps of Backward Elimination (old variables must have p < Sl_stay to stay), then Repeat Step2
#       Step4 - No new Variables can enter and no old variables can exit
#       FIN - Your Model is Ready.
#
#   - `Score Comparision (or All Possible Models)` steps:
#       Step1 - Select a criterion of goodness of fit.
#       Step2 - Construct All Possible Regression Models: 2**n - 1 total combinations
#       Step3 - Select the one with the best criterion
#       FIN - Your Model is Ready.
#     This Approach is too much resource consuming and hence not good for every time use.

# ***** Model-Optimization: Backward Elimination *****
# sklearn.linear_model library accounts for the constant b0, but statsmodels doesn't. So we need to add it manually.
#
import statsmodels.formula.api as sm
features = np.append(arr=np.ones((50, 1)).astype(int), values=features, axis=1)  # axis=1 for column and axis=0 for row.
#
#   Using a threshold value of 0.05 for P-values.
#
#   opt_features = features[:, [0, 1, 2, 3, 4, 5]]      # Step1 - All-in
#   ols_regressor = sm.OLS(endog=output, exog=opt_features).fit()       # Step2 - Fit the model
#   ols_regressor.summary()     # Step3 - Consider the predictor with the highest P-value
#
#   opt_features = features[:, [0, 1, 3, 4, 5]]      # Step4 - Remove the predictor
#   ols_regressor = sm.OLS(endog=output, exog=opt_features).fit()    # Step2
#   ols_regressor.summary()     # Step3
#
#   opt_features = features[:, [0, 3, 4, 5]]        # Step4
#   ols_regressor = sm.OLS(endog=output, exog=opt_features).fit()    # Step2
#   ols_regressor.summary()     # Step3
#
#
# Way-1: Using threshold values ----------------------------------------------
opt_features = features[:, [0, 3, 5]]           # Step4
ols_regressor = sm.OLS(endog=target, exog=opt_features).fit()   # Step2
ols_regressor.summary()     # Step3
opt_features = features[:, [3]]                 # Step4
# Removing not only feature-5 but also the feature-0 as it is the constant value we have added.

# ***** Checking the Results with opt_features *****
opt_training_features = training_features[:, [2]]
opt_testing_features = testing_features[:, [2]]
opt_regressor = LinearRegression()
opt_regressor.fit(opt_training_features, training_target)
opt_predicted_target = opt_regressor.predict(opt_testing_features)
opt_error = abs(testing_target - opt_predicted_target)

# ***** Visualising Results (Optimised) *****
#   - Visualising the Training set results
plt.subplot(121)
plt.scatter(opt_training_features, training_target, color='red')
plt.plot(opt_training_features, opt_regressor.predict(opt_training_features), color='blue')
plt.title('Profit vs R&D-Spend (Training-Set)')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
#   - Visualising the Test set results
plt.subplot(122)
plt.scatter(opt_testing_features, testing_target, color='red')
plt.plot(opt_testing_features, opt_predicted_target, color='blue')
plt.title('Profit vs R&D-Spend (Testing-Set)')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.show()


# Way-2: Using Adjusted-R**2 values ------------------------------------------
opt_features = features[:, [0, 3, 5]]           # Step4
ols_regressor = sm.OLS(endog=target, exog=opt_features).fit()   # Step2
ols_regressor.summary()     # Step3
opt_features = features[:, [0, 3]]                 # Step4
ols_regressor = sm.OLS(endog=target, exog=opt_features).fit()   # Step2
ols_regressor.summary()     # Step3
# Since, the Adjusted-R value drops with the drop of variable no.-5, indicates us that the variable no.-5 is also an
# effective variable (or regressor).

# ***** Checking the Results with opt_features *****
opt_training_features = training_features[:, [2, 4]]
opt_testing_features = testing_features[:, [2, 4]]
opt_regressor = LinearRegression()
opt_regressor.fit(opt_training_features, training_target)
opt_predicted_target = opt_regressor.predict(opt_testing_features)
opt_error2 = abs(testing_target - opt_predicted_target)
