# solver.py
from typing import Any, Dict

import numpy as np
from sklearn.cluster import SpectralClustering


class Solver:
    """
    Optimized spectral clustering solver using sklearn's SpectralClustering.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the spectral clustering problem.

        Parameters
        ----------
        problem : dict
            Must contain:
            - 'similarity_matrix': 2‑D square numpy array
            - 'n_clusters': number of clusters (int > 0)

        Returns
        -------
        dict
            Contains 'labels': numpy array of cluster assignments.
        """
        similarity_matrix = problem["similarity_matrix"]
        n_clusters = problem["n_clusters"]

        # basic validation
        if (
            not isinstance(similarity_matrix, np.ndarray)
            or similarity_matrix.ndim != 2
            or similarity_matrix.shape[0] != similarity_matrix.shape[1]
        ):
            raise ValueError("Invalid similarity matrix")

        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError("Invalid number of clusters")

        n = similarity_matrix.shape[0]
        if n_clusters >= n:
            labels = np.arange(n, dtype=int)
        elif n == 0:
            labels = np.empty((0,), dtype=int)
        else:
            # spectral clustering with precomputed affinity
            model = SpectralClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=42,
                n_init=10,
            )
            try:
                labels = model.fit_predict(similarity_matrix)
            except Exception:
                labels = np.zeros(n, dtype=int)

        return {"labels": labels}