''' Support Vector Machine (SVM) Classification '''

# ***** SVM Classifier Intution *****
# In this, say for example we have two or more different types of categories in which we can classify our new data points.
# So, graphically speaking, we are gonna see that how we can separate the boundaries for these two categories, so that when
# a new point arrives on the basis of decision region it fall, we can predict the category of that point.
#
# Both The Logistic and KNN made the boundaries and regions on that basis to classify the data, where, Logistic classifier
# gives linear boundary and KNN gives non-linear boundary. WhileSVM works on a linearly separable dataset but with a bit
# different functionality (i.e., Mapping to a Higher Dimension).
#
# Now thinking graphically for a linearly separable dataset having two categories, then suppose we separate them with a 
# linear line. Then for SVM, we are gonna find our `Support Vectors` that are two points (in our case, since we are taking 
# two categories) each one representing one of the category. So we are gonna focus on those two points only and they helps
# in deciding the decision boundary region. Now for a good model, we tend to provide the maximum margin possible for both 
# the support vectors from that line (also called as Maximum Margi Hyperplane).
#
# Now, what's so special about SVMs?
#   - Unlike other Ml algo, where we tend to choose the farthest of the two points of each category as the support vectors,
#     that are being located at the center or in a densely populated areas of their categorical regions, here we are gonna 
#     take the ones that are the closest points to the decision boundary in each of the category. So SVM generally have a 
#     thought process of, learning about data points which are least similar in their category are are near to the boundary.
#
# ``` Mapping to a Higher Dimension ```
# This is the scenario when we direclty cannot separate categories on the basis of a linear separator. Say for a 2D plane,
# we are not able to separate data of two categories formin concentric circluar regions. One category has the inner region
# and other category resides in the concentric region then even though we do not have any means of separating them linearly
# in 2D, but if we are gonna plot them in a higher dimension say - 3D, then there may exist a linear separator which can
# linearly separate those two regions (though in this case that linear separator will be a - hyperplane).
#
# But there is a problem with Mapping to a Higher Dimension:
#   - Mapping to a Higher Dimensional Space can be highly compute-intensive. Since we first need to convert the dataset to
#     a higher dimension and then find the linear separator and then bring back all the dataset and the linear separator 
#     (or just the linear separator) back to the initial dimensions - are a work of high computations.
#   - For the solution of the above problem we explore a different approach which is called in mathematics the `kernel` 
#     trick and that approach is going to help us get very similar results but without having to go to a higher dimensional 
#     space.
#
# ``` The Kernel Trick ```
# The most frequently used kernel is the Gaussaian_RBF Kernel. For that we have the following formula:
#   --> K(x, l**i) = e ** (-1*(||x - l|| ** 2) / 2*(sigma**2)) 
# where, l = landmark vector (i.e., the center on the x-y plane directly below the peak that elevates from the x-y plane)
#        x = positon vector on the x-y plane
#        sigma = some predefined constant
# So as we got our above formula, in the explanation we are taking the case of a 2D plane. Then for a 2D plane when we
# get a Gaussian-RBF kernel (i.e, the circular region converging to a higher dimension resulting in the peak), 
# then we choose the landmark point. Then for every new point on that 2D plane we calculate the kernel fucntion, if the
# vector x is far from the vector l, then the value of `x - l` will be large, sqaure of which will be much more larger. 
# And as we know the larger the negative power of e is the more closer the value is to 0 since e**(-inf) = 0.
#
# And if the vector x is near the landmark vector then for that the value of `x - l` will be small, sqaure of which will
# be even smaller. And as we know the more the value of power tends to 0 the value of e raise to that power will converge
# to 1.
#
# Hence with this kernel trick we got our two decision regions separated as it was being separated by a higher dimension
# linear separator in a higher dimensional space.
#
# Now here, the role of sigma is to define how large the circumference of the elevated region is at the x-y plane.
# This is the region that turns out to be decision boundary.
#
# And Hence with the help of SVM we got a Non-Linear Decision boundary.

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

# Fitting SVM classifier to our Training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
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
plt.title('SVM - `Gaussain-RBF` (Training Set)')
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
plt.title('SVM - `Gaussian-RBF` (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()