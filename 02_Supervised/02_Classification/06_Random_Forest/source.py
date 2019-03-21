''' Random Forest Classification '''

# ***** Random-Forest-Regression Intution *****
# (Ensemble Learning Algorithms: It is type of learning in which you put the same algorithm multiple times or combine
#  different algorithms to make something much more powerful than the original one.)
#   - Step1: Pick at random `k` data points from the Training set.
#   - Step2: Build the Decision Tree associated to these `k` data points.
#   - Step3: Choose the number `n` (number of trees you want to build) and repeat steps1 and 2.
#   - Step4: For a new data point, make each one of your `n` trees predict the category of `y` and assign the new data
#            point to category which is predicted for the most number of times.

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

# Fitting Random Forest classifier to our Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=0)
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
plt.title('Random Forest (Training Set)')
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
plt.title('Random Forest (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show() 