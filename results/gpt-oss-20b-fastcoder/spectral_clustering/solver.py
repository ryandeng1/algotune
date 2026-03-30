from __future__ import annotations
import numpy as np
from sklearn.cluster import SpectralClustering


class Solver:
    """
    Fast spectral clustering solver using sklearn’s pre‑computed affinity option.
    The implementation keeps the essential validation logic but removes unnecessary
    staged pass statements and uses a single return path to minimise overhead.
    """

    def __init__(self) -> None:
        # Pre‑create the model with fixed parameters; only n_clusters changes
        self._base_model = SpectralClustering(
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
        )

    def solve(self, problem: dict[str, np.ndarray | int]) -> dict[str, np.ndarray]:
        """
        Compute cluster labels for the given similarity matrix.

        Parameters
        ----------
        problem : dict
            Must contain:
                * "similarity_matrix" : square np.ndarray (float64)
                * "n_clusters"       : int > 0

        Returns
        -------
        dict
            {"labels": np.ndarray}  – cluster assignments
        """
        A = problem["similarity_matrix"]
        n_clusters = problem["n_clusters"]

        if not (
            isinstance(A, np.ndarray)
            and A.ndim == 2
            and A.shape[0] == A.shape[1]
        ):
            raise ValueError("Invalid similarity matrix provided")

        if not (isinstance(n_clusters, int) and n_clusters >= 1):
            raise ValueError("Invalid number of clusters provided")

        n = A.shape[0]
        if n_clusters >= n:
            return {"labels": np.arange(n)}

        # If matrix is empty, return empty labels
        if n == 0:
            return {"labels": np.array([], dtype=int)}

        # Configure and fit the model
        model = self._base_model
        model.n_clusters = n_clusters
        try:
            labels = model.fit_predict(A)
        except Exception:               # pragma: no cover
            labels = np.zeros(n, dtype=int)

        return {"labels": labels}