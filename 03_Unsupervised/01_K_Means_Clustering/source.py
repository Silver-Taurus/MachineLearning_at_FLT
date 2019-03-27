''' K-Means Clustering '''

# ***** K-Means Clustering Intution *****
# Suppose we have data points (or the information on the scatter plot) which can be group together to represent one class
# or category of their own. For those types of information we don't have a predefined value to check our prediction from
# like in case of supervised learning, that is why it called unspuervised learning.
#
# In K-Means Clustering we form the clusters of the data points that the algo predicts should belong to the same class or
# category. 
#
# Now, How the K-Means Clustering works?
#   Step1: Choose the number of K of clusters
#   Step2: Select at random K points, the centroids (not necessarily from your dataset).
#   Step3: Assign each data point to the closest centroid --> That forms K clusters.
#   Step4: Compute and place the new centroid of each cluster.
#   Step5: Re-assign each data point to the newest closest centroid. If any reassignment took place, go to Step4,
#          otherwise, Your Model is Ready!
#
# ``` Random Initialisation Trap ``` 
# Since in the case of our normal K-Means Clustering the output is highly dependent on the initial random centroid points
# So, even if our model gets ready, the clusters that are formed might be different from the true clusters.
#
# ``` K Means++ ```
# In order to overcome the problem of Random Initialisation Trap, we need to implement the K Means++ algorithm, that is
# an algorithm for choosing the initial values (or "seeds") for the K-Means clustering algorithm.
# The exact algorithm is as follows:
#   Step1: Choose one center uniformly at random from among the data points.
#   Step2: For each data point x, compute D(x), the distance between x and the nearest center that has already been chosen.
#   Step3: Choose one new data point at random as a new center, using a weighted probability distribution where a point x 
#          is chosen with probability proportional to D(x)2.
#   Step4: Repeat Steps 2 and 3 until k centers have been chosen.
#   Step5: Now that the initial centers have been chosen, proceed using standard k-means clustering.
#
# So, For us Python libraries will take care of the K Means++ implementation inside the K-Means clustering algorithm.
#
# Now, Choosing the right number of clusters (``` The Elbow Method ```):
#   --> WCSS = sum(distance(Pi, C1) ** 2) + sum(distance(Pi, C2) ** 2) + ...
# where, WCSS is `Within-Cluster Sum of Squares`.
#        Pi = Point in the cluster
#        C1, C2, ... = Clusters
# For choosing the right amount of clusters that we need to provide to the K-Means Clustering algorithm first we calculate
# the WCCS value for some set of K (say for K = 1 to K = 10). After that we plot the graph for WCSS v.s. K values. The part
# of the graph where we find the `elbow` of the graph, we choose that K-value.
# Now, how the WCSS gives the metric for choosing the K-values: When we have a single cluster then the centroid will be at
# center of the dataset plot due to which the distances of the centroid from each point in the cluster will be very large
# and the sqaure of it will be even larger and there summation is also going to be a large value. Hence, as we are going
# on increasing the K-values the value of WCSS will gonna decrease and the point where the large drops in the value of
# WCSS with the unit decrement in K-value converges we got our `elbow` of the graph.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Mall_Customers.csv')

# Making the Information matrix
info = dataset.iloc[:, [3, 4]].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = [KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10,  random_state=0).fit(info).inertia_ for i in range(1, 11)]
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# `So from the elbow method we got our optimal K-value --> 5.`

# Applying K-Means to the dataset
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(info)

# Visualising the Clusters
plt.scatter(info[clusters == 0, 0], info[clusters == 0, 1], s=100, c='red', label='Careful')
plt.scatter(info[clusters == 1, 0], info[clusters == 1, 1], s=100, c='blue', label='Standard')
plt.scatter(info[clusters == 2, 0], info[clusters == 2, 1], s=100, c='green', label='Target')
plt.scatter(info[clusters == 3, 0], info[clusters == 3, 1], s=100, c='grey', label='Careless')
plt.scatter(info[clusters == 4, 0], info[clusters == 4, 1], s=100, c='purple', label='Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow', label='Centroids')
plt.title('Clusters of Clients (K-Means Clustering)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
