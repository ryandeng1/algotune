# solver.py
import numpy as np

class Solver:
    def __init__(self):
        # Use a fixed RandomState for reproducibility
        self.rng = np.random.default_rng(42)

    def _initialize_centroids(self, X, k):
        """Randomly choose k unique data points as initial centroids."""
        n = X.shape[0]
        indices = self.rng.choice(n, size=k, replace=False)
        return X[indices].astype(np.float64)

    def _assign_clusters(self, X, centroids):
        """Assign each point to the nearest centroid."""
        # Compute squared distances (efficient broadcasting)
        diff = X[:, None, :] - centroids[None, :, :]  # shape (n, k, d)
        sq_dist = np.sum(diff * diff, axis=2)        # shape (n, k)
        return np.argmin(sq_dist, axis=1)

    def _update_centroids(self, X, labels, k, eps=1e-12):
        """Update centroids as the mean of assigned points."""
        d = X.shape[1]
        centroids = np.empty((k, d), dtype=np.float64)
        for i in range(k):
            cluster = X[labels == i]
            if cluster.size == 0:
                # Reinitialize empty cluster to a random point
                centroids[i] = X[self.rng.integers(0, X.shape[0])]
            else:
                centroids[i] = cluster.mean(axis=0)
        return centroids

    def solve(self, problem: dict[str, any]) -> list[int]:
        """
        Implements a fast deterministic Lloyd k-means algorithm.
        It iterates until convergence or a maximum number of iterations.
        """
        X = np.asarray(problem["X"], dtype=np.float64)
        k = int(problem["k"])
        n, d = X.shape

        # Edge cases
        if k <= 0:
            return [0] * n
        if k == 1:
            return [0] * n

        centroids = self._initialize_centroids(X, k)
        labels = np.empty(n, dtype=np.int32)

        for _ in range(15):  # 15 iterations is usually enough for small datasets
            new_labels = self._assign_clusters(X, centroids)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            centroids = self._update_centroids(X, labels, k)

        # Ensure labels are in the range [0, k-1]
        return labels.tolist()
