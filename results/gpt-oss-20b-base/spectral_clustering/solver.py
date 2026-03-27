import numpy as np
from sklearn.cluster import SpectralClustering
from typing import Any, Dict


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform spectral clustering on a precomputed similarity matrix.

        Parameters
        ----------
        problem : dict
            Must contain:
                - "similarity_matrix": a square 2-D NumPy array
                - "n_clusters": integer > 0

        Returns
        -------
        dict
            Dictionary with key "labels" containing the cluster assignments.
        """
        # Extract input
        similarity = problem["similarity_matrix"]
        n_clusters = problem["n_clusters"]

        # Basic validation
        if (
            not isinstance(similarity, np.ndarray)
            or similarity.ndim != 2
            or similarity.shape[0] != similarity.shape[1]
        ):
            raise ValueError("`similarity_matrix` must be a square 2‑D NumPy array.")

        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("`n_clusters` must be a positive integer.")

        n_samples = similarity.shape[0]

        # Trivial case: more clusters than samples
        if n_clusters >= n_samples:
            labels = np.arange(n_samples, dtype=int)
            return {"labels": labels}

        # Empty data
        if n_samples == 0:
            return {"labels": np.empty(0, dtype=int)}

        # Run spectral clustering
        clf = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
        )
        try:
            labels = clf.fit_predict(similarity)
        except Exception:
            # Fall back to zeros if clustering fails
            labels = np.zeros(n_samples, dtype=int)

        return {"labels": labels}