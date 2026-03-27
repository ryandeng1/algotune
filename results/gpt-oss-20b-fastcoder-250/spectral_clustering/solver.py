import numpy as np
from sklearn.cluster import SpectralClustering
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the spectral clustering problem using sklearn's SpectralClustering.

        Parameters
        ----------
        problem : dict
            Must contain keys:
            - "similarity_matrix": square numpy array of shape (n_samples, n_samples)
            - "n_clusters": int, the desired number of clusters

        Returns
        -------
        dict
            {"labels": np.ndarray} where `labels[i]` is the cluster assignment of sample i.
        """
        similarity_matrix = problem["similarity_matrix"]
        n_clusters = problem["n_clusters"]

        # --- Input validation -------------------------------------------------------
        if (
            not isinstance(similarity_matrix, np.ndarray)
            or similarity_matrix.ndim != 2
            or similarity_matrix.shape[0] != similarity_matrix.shape[1]
        ):
            raise ValueError("`similarity_matrix` must be a square 2‑D numpy array.")
        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError("`n_clusters` must be a positive integer.")

        n_samples = similarity_matrix.shape[0]

        # --- Edge cases -------------------------------------------------------------
        if n_samples == 0:
            return {"labels": np.array([], dtype=int)}

        if n_clusters >= n_samples:
            # Each sample gets its own cluster – trivial solution
            return {"labels": np.arange(n_samples)}

        # --- Main computation -------------------------------------------------------
        # The clustering procedure is deterministic via `random_state`.
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
        )
        try:
            labels = model.fit_predict(similarity_matrix)
        except Exception:
            # Fallback to zero labels in case of a numerical failure
            labels = np.zeros(n_samples, dtype=int)

        return {"labels": labels}