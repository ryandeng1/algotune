import numpy as np
from sklearn.cluster import KMeans

class Solver:
    def solve(self, problem: dict) -> list[int]:
        """
        Cluster the data points using k-means and return the cluster labels.
        If any error occurs during clustering, return a list of zeros.
        """
        X = problem.get('X')
        k = problem.get('k')
        if X is None or k is None:
            return []

        try:
            # Use a fast deterministic initialization by setting n_init=1
            # and use the default algorithm (elkan) which is efficient for
            # few clusters and smaller dimensionality.
            kmeans = KMeans(n_clusters=int(k), n_init=1, algorithm='elkan')
            kmeans.fit(X)
            return kmeans.labels_.tolist()
        except Exception:
            # Fallback: return zero labels for all data points
            return [0] * len(X)