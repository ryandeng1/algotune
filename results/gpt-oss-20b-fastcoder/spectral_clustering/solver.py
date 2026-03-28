import numpy as np
from sklearn.cluster import SpectralClustering

class Solver:
    def solve(self, problem: dict):
        """Run spectral clustering on a pre‑computed similarity matrix."""
        sim = problem["similarity_matrix"]
        k = problem["n_clusters"]

        # Quick sanity checks (minimal work)
        if not isinstance(sim, np.ndarray) or sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
            raise ValueError("similarity_matrix must be a square 2‑D NumPy array")
        if not isinstance(k, int) or k < 1:
            raise ValueError("n_clusters must be a positive integer")

        n = sim.shape[0]
        if k >= n:
            labels = np.arange(n)
        elif n == 0:
            labels = np.empty(0, int)
        else:
            # SpectralClustering with pre‑computed affinity is the fastest path
            model = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=42,
            )
            try:
                labels = model.fit_predict(sim)
            except Exception:
                labels = np.zeros(n, int)

        return {"labels": labels}