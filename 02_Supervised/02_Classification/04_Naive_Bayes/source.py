''' Naive Bayes Classification '''

# ***** Naive-Bayes Classifier Intitution *****
#
# The Naive Bayes Classifier as it's name suggests is based on the Naive Bayes Theorem, that is:
#   --> P(A|B) = P(B|A) * P(A) / P (B)
# where, A and B are two events.
#
# Now the Plan of Attack is:
#   - First let us assume X as the features of the data points and there are two actions or results that can be taken by
#     that data points (say actions, A1 and A2). Considering the data point as a person having X features (say, Age and
#     Salary) the actions that he can take are A1 (Walks to office) and A2 (drives to office).
#   - The following are the steps to be taken:
#       Step1: Calculate -->  P(Walks|X) = P(X|Walks) * P(Walks) / P(X)
#               where, P(Walks) = Prior Probability
#                      P(X) = Marginal Likelihood
#                      P(X|Walks) = Likelihood
#                      P(Walks|X) = Posterior Probability
#       Step2: Calculate -->  P(Drives|X) = P(X|Drives) * P(Drives) / P(X)
#       Step3: P(Walks|X) v.s. P(Drives|X) - This is the last step where we choose where to put the new data point.
#
# Why is it called `Naive`?
#   - The answer is because of the concept used in Naive Bayes is Bayes Theorem which requires some assumptions for it's
#     foundation and some time these assumptions are not true. For Bayes theorem the feature X1 and X2 should be
#     independent as they are collectively taken as a unit X and thus they cannot show dependence on each other in
#     sub-parts.
#   
# What happens when we have more than 2 classes?
#   - When we have two classes we calculate for the one class and then for the other class we subtract the result from 1.
#     But when we have more than 2 classes (say, 3 for example), then we are gonna calculate for the two classes and then
#     subtract their sum from 1 and get the result for the third class.
#
# This is also a Non-Linear Classification. 

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

# Fitting Naive Bayes classifier to our Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
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
plt.title('Naive Bayes (Training Set)')
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
plt.title('Naive Bayes (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show() 