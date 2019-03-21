''' SVR '''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Making feature matrix and target matrix (for SVR)
features = dataset.iloc[:, [1]].values
target = dataset.iloc[:, [-1]].values

# Feature Scaling
#   - Many of the Python ML libraries take care of it, so we don't need to do it, for the SVR we have to do it.
from sklearn.preprocessing import StandardScaler
features_ssc = StandardScaler()
target_ssc = StandardScaler()
features = features_ssc.fit_transform(features)
target = target_ssc.fit_transform(target)

# Fitting the SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(features, target)

# Predicting a new result
#   - predict() requires a matrix
feature_value = np.array([[6.5]])
feature_value = features_ssc.transform(feature_value)
predicted_target = regressor.predict(feature_value)

# Prediction value
predicted_target_value = target_ssc.inverse_transform(predicted_target)

# Visualising the SVR results
plt.scatter(features, target, color='red')
plt.plot(features, regressor.predict(features), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Here, in the SVR there are some penalty parameters according to which the CEO is outlier.
