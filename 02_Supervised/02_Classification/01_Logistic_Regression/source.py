''' Logistic Regression Classification '''

# ***** Logistic-Regression Intution *****
# This is a regression type algo that we use for classifcation problems (i.e., 0 or 1), either we perform this action or
# that action. For this we calculate the probability or likelihood of happening any event on the basis of the trained data.
# In this, the approach will be similar to that of the Linear Regression but since we have the output 0 or 1 we will
# be using a sigmoid function and thus we will get our `Logistic-Regression`.
#
# As we know the Simple-Linear Regression formula,
#   --> y = b0 + b1*x
#
# Then, using sigmoid function: 
#   --> p = 1 / 1 + e^(-y)
# where, p is probability.
#
# We got our Logistic-Regression formula,
#   --> ln(p / 1-p) = b0 + b1*x
#
# So the line we got from this is similar to that of Linear Regression line but just the upper bound and lower bound are
# fixed using the sigmoid function. So we got a graph for p vs x and hence for every value of x we got a probability value.
# So, we can set a p-value (say 0.5), then anything on or above it will give you 1 (or yes) as an output and anything below
# it we give you 0 (or no) as an output.
#
# Hence, the logistic regression is a linear classifier.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# Making the feature matrix and target vector
features = dataset.iloc[:, [2, 3]].values
target = dataset.iloc[:, 4].values

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
trained_features, test_features, trained_target, test_target = train_test_split(features, target, test_size=0.25, 
                                                                                random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
feature_ssc = StandardScaler()
trained_features = feature_ssc.fit_transform(trained_features)
test_features = feature_ssc.transform(test_features)

# Fitting Logistic Regression classifier to our Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(trained_features, trained_target)

# Predicting the Test set results
predicted_target = classifier.predict(test_features)

# Making the `Confusion Matrix`
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_target, predicted_target)

# Visualising the Training Set Results
x_set, y_set = trained_features, trained_target
x1, x2 = np.meshgrid(np.arange(min(x_set[:, 0]) - 1, max(x_set[:, 0]) + 1, 0.01), 
                     np.arange(min(x_set[:, 1]) - 1, max(x_set[:, 1]) + 1, 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), alpha=0.75, 
             cmap=ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logistic Regression (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test Set Results
x_set, y_set = test_features, test_target
x1, x2 = np.meshgrid(np.arange(min(x_set[:, 0]) - 1, max(x_set[:, 0]) + 1, 0.01), 
                     np.arange(min(x_set[:, 1]) - 1, max(x_set[:, 1]) + 1, 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), alpha=0.75, 
             cmap=ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logistic Regression (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()