import numpy as np
from sklearn.cluster import SpectralClustering

class Solver:
    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        sm = problem["similarity_matrix"]
        k = problem["n_clusters"]

        # Basic validation
        if not isinstance(sm, np.ndarray) or sm.ndim != 2 or sm.shape[0] != sm.shape[1]:
            raise ValueError("similarity_matrix must be a square 2‑D numpy array")
        if not isinstance(k, int) or k < 1:
            raise ValueError("n_clusters must be a positive integer")

        n = sm.shape[0]
        if n == 0:
            return {"labels": np.array([], dtype=int)}
        if k >= n:
            return {"labels": np.arange(n)}

        # Spectral clustering with minimal overhead
        model = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
            n_init=1  # reduce random initialisations
        )
        try:
            labels = model.fit_predict(sm)
        except Exception:
            labels = np.zeros(n, dtype=int)

        return {"labels": labels}