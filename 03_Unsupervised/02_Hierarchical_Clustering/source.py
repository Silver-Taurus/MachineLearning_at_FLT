''' Hierarchical Clustering '''

# ***** Hierarchical Clustering Intution *****
# Suppose we have data points (or the information on the scatter plot) which can be group together to represent one class
# or category of their own. Now grouping those data points in the clusters is called Clustering.
#
# Note: Hierarchical Clustering is of two types - Agglomerataive and Divisive.
#       Here, Agglomerative is bottom-up approach where as Divisive is the top-down approach.
#
# Now, Here we are going to deal with Agglomerative approach.
#
# Now Steps in the Agglomerative HC:
#   Step1: Make eash data point a single-point cluster --> That forms N clusters.
#   Step2: Take the two closest data points and make them one cluster --> Thata forms N-1.
#   Step3: Take the two closest clusters and make them one cluster --> That forms N-2.
#   Step4: Repeat Step3 until there is only 1 cluster.
#
# Now here in the Step3 we are not telling about some euclidean distance of some points but the proximity of two clusters 
# itself. For measuring the distance between two clusters, we can have following approaches:
#   - Distance between Closest Points
#   - Distance between Furthest Points
#   - Average Distance
#   - Distance between Centroids
#
# Now Out of all the Clusters that are formed at every level, the main question is to find the optimal number of clusters
# so that we can select the level where the optimal clustering takes place. Here we got `Dendograms`.
#
# How Do Dendograms Work?
# Suppose we have 6 points in our dataset thar are being plotted on the scatter plot in 2D. Then, firstly we are going to
# make each of the 6 points as individual clusters. Then take two closest clusters and forms one cluster. Say, in our case,
# P2 and P3 are formed as single cluster. Then in our Dendogram, we joins the P2 (2 on the x-axis) and P3 (3 on the x-axis)
# by a straight line but at `d` distance above the both points P2 and P3 where, d is the euclidean distance between the two
# points and the larger the distance the more the dissimilarity is shown.
# Now we are gonna repeat the same process. Suppose in the next case we are going to connect the cluster P1 and cluster P2-P3
# then the line will starts from the center of P2-P3 connected line and for P1 it starts from the x-axis till the `d` distance
# above the x-axis. So, we are gonna repeat this process till we got our final one cluster.
#
# So, the Dendograms work as the memory of Hierarchical Clusterings.
#
# Now, How do we use Dendograms to get the Clusters from HC?
# Since we know the vertical distance of the lines for different points shows the dissimilarity of them with each other.
# Hence, we can take some standard dissimilarity threshold according to which we separates hierarchical cluster into 
# different clusters. This value can be thought of as a horizontal line which cuts the longest vertical line (that were, 
# d vertical distance for each points when they were combining into hierarchical clusters), the horizontal line will cut 
# this longest vertical line between the place where it is intercepting by an extended line a real level line.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Mall_Customers.csv')

# Making the Information Matrix
info = dataset.iloc[:, [3, 4]].values

# Using the dendrograms to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(info, method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Eucledian distances')
plt.show()
 
# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
clusters = hc.fit_predict(info)

# Visualising the Clusters
plt.scatter(info[clusters == 0, 0], info[clusters == 0, 1], s=100, c='red', label='Careful')
plt.scatter(info[clusters == 1, 0], info[clusters == 1, 1], s=100, c='blue', label='Standard')
plt.scatter(info[clusters == 2, 0], info[clusters == 2, 1], s=100, c='green', label='Target')
plt.scatter(info[clusters == 3, 0], info[clusters == 3, 1], s=100, c='grey', label='Careless')
plt.scatter(info[clusters == 4, 0], info[clusters == 4, 1], s=100, c='purple', label='Sensible')
plt.title('Clusters of Clients (Agglomerative H.C Clustering)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
