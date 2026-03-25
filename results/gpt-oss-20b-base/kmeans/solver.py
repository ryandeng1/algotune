# solver.py
from typing import Any, List
import numpy as np
from sklearn.cluster import KMeans

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[int]:
        """
        Solve k-means clustering for the given problem.
        """
        X = problem["X"]
        k = problem["k"]
        # Use sklearn's KMeans with default parameters for speed.
        kmeans = KMeans(n_clusters=k, n_init=1, max_iter=300, algorithm="auto", tol=1e-4)
        kmeans.fit(X)
        return kmeans.labels_.tolist()
