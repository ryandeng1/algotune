from typing import Any
import numpy as np
import sklearn.cluster

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[int]:
        X = problem["X"]
        k = problem["k"]
        # Fast KMeans implementation from scikit-learn
        kmeans = sklearn.cluster.KMeans(n_clusters=k, n_init=1, algorithm="elkan")
        kmeans.fit(X)
        return kmeans.labels_.tolist()