import numpy as np
from numba import njit

class Solver:
    def __init__(self):
        pass
    
    @njit
    def _assign(self, X, centroids):
        n, d = X.shape
        k = centroids.shape[0]
        labels = np.zeros(n, dtype=np.int32)
        for i in range(n):
            min_dist = np.inf
            min_idx = 0
            for j in range(k):
                dist = np.sum((X[i] - centroids[j]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = j
            labels[i] = min_idx
        return labels
    
    @njit
    def _update(self, X, labels, k):
        n, d = X.shape
        centroids = np.zeros((k, d), dtype=np.float64)
        for i in range(k):
            mask = (labels == i)
            if np.any(mask):
                centroids[i] = np.mean(X[mask], axis=0)
            else:
                centroids[i] = X[0]
        return centroids
    
    def solve(self, problem: dict[str, Any]) -> list[int]:
        X = problem["X"]
        k = problem["k"]
        
        X = np.array(X, dtype=np.float64)
        
        centroids = self._kmeans_init(X, k)
        
        n = X.shape[0]
        max_iter = 300
        for _ in range(max_iter):
            labels = self._assign(X, centroids)
            new_centroids = self._update(X, labels, k)
            if np.all(np.isclose(centroids, new_centroids)):
                break
            centroids = new_centroids
        
        return labels.tolist()
    
    def _kmeans_init(self, X, k):
        n = X.shape[0]
        centroids = np.zeros((k, X.shape[1]), dtype=np.float64)
        idx = np.random.randint(n)
        centroids[0] = X[idx]
        
        for i in range(1, k):
            D = np.zeros(n)
            for j in range(i):
                dist = np.sum((X - centroids[j]) ** 2, axis=1)
                D = np.maximum(D, dist)
            P = D / np.sum(D)
            idx = np.random.choice(n, p=P)
            centroids[i] = X[idx]
        return centroids
