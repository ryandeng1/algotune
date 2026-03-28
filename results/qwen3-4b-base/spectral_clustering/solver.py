from typing import Any
import numpy as np
from sklearn.cluster import SpectralClustering


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        similarity_matrix = problem["similarity_matrix"]
        n_clusters = problem["n_clusters"]
        
        # Validate input
        if not isinstance(similarity_matrix, np.ndarray) or similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
            raise ValueError("Invalid similarity matrix provided.")
        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError("Invalid number of clusters provided.")
        
        n_samples = similarity_matrix.shape[0]
        
        if n_clusters >= n_samples:
            labels = np.arange(n_samples)
        elif n_samples == 0:
            labels = np.array([], dtype=int)
        else:
            model = SpectralClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=42,
            )
            try:
                labels = model.fit_predict(similarity_matrix)
            except Exception:
                labels = np.zeros(n_samples, dtype=int)
        
        return {"labels": labels}