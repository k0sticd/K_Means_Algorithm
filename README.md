# K_Means_Algorithm
Implementation of the basic K-means algorithm

Funcionalities:

Implementation of a basic k-Means algorithm, with attribute normalization and stopping criteria for optimization, as well as manual selection of the number of clusters. The code is organized through the methods of the class.

Added functionality: allowing the definition of attribute weights (importance) and compute similarity taking these weights into account.

Added functionality: allowing changing the distance measure, so that in addition to the Euclidean metric, other metrics, such as the city-block metric, can be used. 3 additional distance measures added.

Added functionality: allowing the entire clustering process to be automatically restarted N times, and finally report the best clustering model from all attempts.

Added functionality: improving centroid initialization so that instead of random initialization, centroids are selected to be as far apart as possible.

Added functionality: When displaying the results, warn the user if there are clusters that are poorly represented by their centroids (where the distance is greater than an acceptable threshold), or if there are clusters that are too similar.

Added functionality: allowing the algorithm to automatically select the optimal number of clusters using the silhouette index.
