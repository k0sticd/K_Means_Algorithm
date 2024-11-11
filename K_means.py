import numpy as np
from sklearn.metrics import silhouette_score


class KMeans:
    def __init__(self, k=3, max_iter=50, n_init=10, distance_metric='euclidean', init='k-means++', weights=None):
        """
        Constructor for the KMeans class.
        
        Args:
        k (int): Number of clusters.
        max_iter (int): Maximum number of iterations.
        n_init (int): Number of initializations for k-means.
        distance_metric (str): Distance metric ('euclidean', 'cityblock', etc.).
        init (str): Method for initializing centroids ('k-means++' or 'random').
        weights (array-like): Attribute weights.
        """
        self.k = k
        if self.k <= 1:
            raise ValueError("The number of clusters 'k' must be greater than 1.")
        self.max_iter = max_iter
        self.n_init = n_init
        self.distance_metric = distance_metric
        self.init = init
        self.weights = weights
        self.centroids = None
        self.labels_ = None
        self.mean_ = None
        self.std_ = None
        self.check_clusters_during_learn = True

    def _initialize_centroids(self, data):
        """
        Initializes cluster centers based on the selected method.

        """
        
        if self.init == 'k-means++':
            n_samples, n_features = data.shape
            if self.k > n_samples:
                raise ValueError(f"The number of clusters 'k' ({self.k}) cannot be greater than the number of samples ({n_samples}) in the data.")
            centroids = np.zeros((self.k, n_features))
            centroids[0] = data[np.random.choice(n_samples)]
            #centroids[0] = data.sample().values[0]
            for i in range(1, self.k):
                distances = np.min(self._compute_distance(data, centroids[:i]), axis=1)
                probabilities = distances / np.sum(distances)
                centroids[i] = data[np.random.choice(n_samples, p=probabilities)]
                #centroids[i] = data.sample(weights=probabilities).values[0]
            return centroids
        elif self.init == 'random':
            if self.k > data.shape[0]:
                raise ValueError(f"The number of clusters 'k' ({self.k}) cannot be greater than the number of samples ({data.shape[0]}) in the data.")
            return data[np.random.choice(data.shape[0], self.k, replace=False)] #data.sample(self.k).values
        elif self.init == 'farther':
            n_samples, n_features = data.shape
            centroids = np.zeros((self.k, n_features))
            # Select the first centroid randomly
            centroids[0] = data[np.random.choice(n_samples)]
            for i in range(1, self.k):
                # Calculate the distance of each sample from the nearest centroid
                distances = np.min(self._compute_distance(data, centroids[:i]), axis=1)
                # Select the centroid that is farthest from the existing centroids
                farthest_point_index = np.argmax(distances)
                centroids[i] = data[farthest_point_index]
            return centroids
        else:
            raise ValueError(f"Unrecognized initialization method: {self.init}")

    def _apply_weights(self, data):
        """
        Applies attribute weights to the data if specified.

        """
        if self.weights is not None:
            if len(self.weights) != data.shape[1]:
                raise ValueError(f"The number of weights ({len(self.weights)}) does not match the number of attributes ({data.shape[1]}) in the data.")
            return data * self.weights
        return data

    def _normalize_data(self, data):
        """
        Normalizes the data using the mean and standard deviation.

        """
        self.mean_ = data.mean()
        self.std_ = data.std()
        return (data - self.mean_) / self.std_

    def _compute_distance(self, data, centroids):
        """
        Computes distances between data and centroids using the selected distance measure.

        """
        if self.distance_metric == 'euclidean':
            distances = np.zeros((data.shape[0], centroids.shape[0]))
            for i in range(centroids.shape[0]):
                distances[:, i] = np.sqrt(np.sum((data - centroids[i]) ** 2, axis=1))
        elif self.distance_metric == 'cityblock':
            distances = np.zeros((data.shape[0], centroids.shape[0]))
            for i in range(centroids.shape[0]):
                distances[:, i] = np.sum(np.abs(data - centroids[i]), axis=1)
        elif self.distance_metric == 'minkowski':
            p = 3  # P parameter for Minkowski metric
            distances = np.zeros((data.shape[0], centroids.shape[0]))
            for i in range(centroids.shape[0]):
                distances[:, i] = np.power(np.sum(np.abs(data - centroids[i]) ** p, axis=1), 1/p)
        elif self.distance_metric == 'chebyshev':
            distances = np.zeros((data.shape[0], centroids.shape[0]))
            for i in range(centroids.shape[0]):
                distances[:, i] = np.max(np.abs(data - centroids[i]), axis=1)
        else:
            raise ValueError(f"Unrecognized distance measure: {self.distance_metric}")
        return distances

    def learn(self, data):
        """
        Trains the KMeans model based on the data.

        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be in the format of a numpy ndarray.")
        
        if self.k >= data.shape[0]:
            raise ValueError(f"The number of clusters 'k' ({self.k}) cannot be greater than or equal to the number of samples ({data.shape[0]}).")
        
        data = self._normalize_data(data)
        if self.weights is not None:
            data = self._apply_weights(data)

        best_centroids = None
        best_labels = None
        best_score = -np.inf

        for _ in range(self.n_init):
            centroids = self._initialize_centroids(data)
            for _ in range(self.max_iter):
                distances = self._compute_distance(data, centroids)
                labels = np.argmin(distances, axis=1)
                new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(self.k)])
                
                if np.all(centroids == new_centroids):
                    break
                centroids = new_centroids

            score = silhouette_score(data, labels)
            if score > best_score:
                best_score = score
                best_centroids = centroids
                best_labels = labels

        self.centroids = best_centroids
        self.labels_ = best_labels
        
        if self.check_clusters_during_learn:
            self.check_clusters(data)
            
        print(f"Centroids: {self.centroids[:15]}")


    def transform(self, data):
        """
        Predicts clusters for new data.

        """
        if self.centroids is None:
            raise RuntimeError("The model is not trained. Call 'learn' before transformation.")
        
        data = (data - self.mean_) / self.std_
        if self.weights is not None:
            data = self._apply_weights(data)
        distances = self._compute_distance(data, self.centroids)
        labels = np.argmin(distances, axis=1)
        print(f"Labels: {labels}")
    
    def check_clusters(self, data):
        """
        Analyzes clusters to determine if they are poorly represented or too similar.
        """
        if self.centroids is None or self.labels_ is None:
            raise RuntimeError("The model is not trained. Call 'learn' before analyzing clusters.")
    
        distances = self._compute_distance(self.centroids, self.centroids)
        np.fill_diagonal(distances, np.inf)  # Ignore distances center-centroid of the cluster itself
    
        # Analyze poorly represented clusters
        for i in range(self.k):
            centroid = self.centroids[i]
            cluster_data = data[self.labels_ == i]  # Extract data for cluster i
            if cluster_data.shape[0] == 0:
                print(f"Cluster {i} is empty.")
                continue
            within_cluster_distances = np.linalg.norm(cluster_data - centroid, axis=1)
            mean_distance = np.mean(within_cluster_distances)
            std_distance = np.std(within_cluster_distances)
            
            print(f"Cluster {i} - Mean distance: {mean_distance}, Standard deviation: {std_distance}")
        
            if mean_distance > 3 * std_distance:
                print(f"Cluster {i} is poorly represented.")
            

        # Analyze similar clusters
        for i in range(self.k):
            for j in range(i + 1, self.k):
                if distances[i, j] < 1e-5:  # If clusters are very similar
                    print(f"Cluster {i} and Cluster {j} are too similar.")

    def find_optimal_k(self, data, max_k=10):
        """
        Finds the optimal number of clusters using the silhouette index.

        """
    
        if max_k < 2:
            raise ValueError("max_k must be greater than 1.")
    
        best_k = self.k
        best_score = -np.inf
        
        original_check_clusters_during_learn = self.check_clusters_during_learn
        self.check_clusters_during_learn = False  # Disable check_clusters during optimal number of clusters search
    
        original_k = self.k  # Save the original number of clusters
    
        for k in range(2, max_k + 1):
            self.k = k  # Temporarily set the current number of clusters
            self.learn(data)  # Train the model with the current number of clusters
            labels = self.labels_
            score = silhouette_score(data, labels)
            if score > best_score:
                best_score = score
                best_k = k
    
        self.k = original_k  # Restore the original number of clusters
        self.check_clusters_during_learn = original_check_clusters_during_learn  # Restore original value
        return best_k

# TEST
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('boston.csv')
X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
X_train = data.values # We assume that all attributes in the CSV file represent data for clustering
X_test = data.values

# Creating an instance of the KMeans class
kmeans = KMeans(max_iter=3, n_init=4, distance_metric='euclidean', init='k-means++', k=3)

#optimal_k = kmeans.find_optimal_k(X_train, max_k=10)
#print(f'Optimal number of clusters is: {optimal_k}')

# Model training
kmeans.learn(X_train)
kmeans.transform(X_test)