''' K-Nearest Neighbours (KNN) Classification'''

# ***** KNN Classifier Intution *****
# In this, say for example we have two or more different types of categories in which we can classify our new data points.
# So, graphically speaking, if we got a new point, then we are gonna see, that which category is the nearest neighbour
# to that point.
#
# Following steps are taken to do that:
#   Step1: Choose the number K of neighbours
#   Step2: Take the K nearest neighbours of the new data point, according to the Euclidean distance in our case.
#   Step3: Among those K neighbours, count the number of data points in each category.
#   Step4: Assign the new data point to the category where you counted the most neighbours and Your Model is ready!
#
# KNN is a Non-Linear Classifier.

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

# Fitting KNN classifier to our Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
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
plt.title('KNN (Training Set)')
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
plt.title('KNN (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()