import numpy as np
from typing import Any

class Solver:
    def __init__(self):
        # PageRank damping factor
        self.alpha = 0.85
        # Maximum number of iterations
        self.max_iter = 100
        # Convergence tolerance
        self.tol = 1e-6

    def _build_transition(self, adj_list):
        """
        Build the column‑stochastic transition matrix for the graph.
        Dangling nodes (no out‑links) are treated as nodes pointing to all
        nodes uniformly.
        """
        n = len(adj_list)
        if n == 0:
            return np.empty((0, 0))
        # Start with a zero matrix
        M = np.zeros((n, n), dtype=np.float64)
        for src, neighbors in enumerate(adj_list):
            if neighbors:  # regular node
                prob = 1.0 / len(neighbors)
                M[neighbors, src] = prob
        # Handle dangling nodes: send probability uniformly to all nodes
        dangling = np.where(M.sum(axis=0) == 0)[0]
        if dangling.size:
            M[:, dangling] = 1.0 / n
        return M

    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, list[float]]:
        """
        Calculates the PageRank scores for the graph using a custom
        efficient implementation with NumPy instead of NetworkX.
        """
        adj_list = problem["adjacency_list"]
        n = len(adj_list)

        # Special cases
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}

        # Build transition matrix (column‑stochastic)
        M = self._build_transition(adj_list)

        # Teleportation vector (uniform distribution)
        v = np.full(n, 1.0 / n, dtype=np.float64)

        # Start with a uniform rank vector
        r = v.copy()

        tol = self.tol
        alpha = self.alpha
        max_iter = self.max_iter

        for _ in range(max_iter):
            r_next = alpha * M @ r + (1 - alpha) * v
            # Check convergence
            if np.linalg.norm(r_next - r, ord=1) < tol:
                r = r_next
                break
            r = r_next

        return {"pagerank_scores": r.tolist()}