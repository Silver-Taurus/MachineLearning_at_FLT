''' Decision Tree Classification '''

# ***** Decision-Tree-Classification Intution *****
# (CART: Classification and Regression Trees)
#   - We have two types of Tree algos in ML:
#       - Classification Trees
#       - Regression Trees
#
#   - Here, we are going to use the Classification Trees:
#       - Here, we talk about the prediction of categorical data using the Decison Trees. Here, the information gets 
#         splitted on the basis of algo `information entropy` and stops dividing when there is no more information left 
#         to be get splitted and hence we get the optimal splits and the terminal leaves. This is the concept of `Tree`.
#       - Now, how the `Decision` works is: Suppose we have 4 optimal splits for an information having two features
#         x and y and a dependent variable (in a 2D - scatter plot), say:
#           - 1st split: y = 60
#           - 2nd split: x = 50
#           - 3rd split: x = 70
#           - 4th split: y = 20
#         So, now we got out terminal leaves. Now, whenever we got a value (say, X(x0, y0)), we will check:
#                                   y0 < 60
#                                      /\
#                                (Yes)/  \(No)
#                                    /    \
#                                   /      \
#                              x0 < 70    x0 < 50
#                              /\               /\
#                        (Yes)/  \(No)         /  \
#                            /    \      (Yes)/    \(No)
#                           /      \         /      \
#                        [C-0]    y0 < 20  [C-1]    [C-0]
#                                   /\
#                                  /  \
#                                 /    \
#                                /      \
#                             [C-0]   [C-1]
#
#   - So from the above representation of the Decison-Tree, we get the five Terminal leaves which helps in
#     predicting the output. So how we are going to predict the output is - whenever a point falls in a terminal leaf 
#     region, then the point is categorised as of the same class as that of the region it falls. 
#
# This is ofcourse a Non-Linear Classification.

# Decision Trees on their own maybe are not as powerful as other Ml algos but when used with other algos they perform
# some very high level predictions and recognitions. Some of the other algos with which they are being used are -
# Random Forest, Gradient Boosting, etc. and the fields in which it is used includes the Facial Recognition tasks,
# game-play to detect the movement of our hands, etc.

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

# Fitting Decision Tree classifier to our Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
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
plt.title('Decision Tree (Training Set)')
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
plt.title('Decision Tree (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show() 