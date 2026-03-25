import numpy as np
from sklearn.cluster import SpectralClustering

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        """Spectral clustering using pre‑computed similarity matrix.

        Parameters
        ----------
        problem : dict
            Must contain keys:
                - "similarity_matrix": a 2‑D NumPy array (n_samples × n_samples)
                - "n_clusters": int, desired number of clusters

        Returns
        -------
        dict
            {"labels": labels_array, "n_clusters": k}
        """
        # Extract inputs
        S = problem["similarity_matrix"]
        k = problem["n_clusters"]

        # Validate inputs (small checks – most are guaranteed by task description)
        if not isinstance(S, np.ndarray) or S.ndim != 2 or S.shape[0] != S.shape[1]:
            raise ValueError("similarity_matrix must be a square NumPy array")
        if not isinstance(k, int) or k < 1:
            raise ValueError("n_clusters must be a positive integer")

        n = S.shape[0]

        # Handle degenerate cases
        if n == 0:
            return {"labels": np.array([], dtype=int), "n_clusters": k}
        if k >= n:
            return {"labels": np.arange(n, dtype=int), "n_clusters": k}

        # Spectral clustering with precomputed affinity
        model = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            n_init=10,          # default
            random_state=42,
        )
        labels = model.fit_predict(S)

        return {"labels": labels, "n_clusters": k}
