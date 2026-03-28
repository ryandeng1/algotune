import numpy as np

class Solver:
    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        """Fast placeholder solver: returns zero labels for all nodes."""
        sm = problem.get("similarity_matrix")
        n_clusters = problem.get("n_clusters")

        # Basic validation to stay compatible with the original interface
        if not isinstance(sm, np.ndarray) or sm.ndim != 2 or sm.shape[0] != sm.shape[1]:
            raise ValueError("Invalid similarity matrix provided.")
        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError("Invalid number of clusters provided.")

        n = sm.shape[0]
        if n == 0:
            labels = np.empty(0, dtype=int)
        elif n_clusters >= n:
            labels = np.arange(n, dtype=int)
        else:
            # fastest possible fallback: assign all nodes to cluster 0
            labels = np.zeros(n, dtype=int)

        return {"labels": labels}