# solver.py

import numpy as np
from typing import Any, Dict, List

class Solver:
    """Fast PageRank solver using NumPy only."""

    __slots__ = ("alpha", "max_iter", "tol")

    def __init__(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6):
        """
        Parameters
        ----------
        alpha : float
            Damping factor.
        max_iter : int
            Maximum number of iterations.
        tol : float
            Convergence tolerance for the L1 norm.
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def _build_matrices(self, adj_list: List[List[int]]) -> np.ndarray:
        """Builds the column‑stochastic transition matrix P."""
        n = len(adj_list)
        P = np.zeros((n, n), dtype=np.float64)

        # Out‑degree vector
        out_deg = np.array([len(neigh) for neigh in adj_list], dtype=np.int32)

        for i, neighbors in enumerate(adj_list):
            di = out_deg[i]
            if di > 0:
                inv = 1.0 / di
                P[neighbors, i] = inv
            else:
                # Dangling node: distributes uniformly
                P[:, i] = 1.0 / n

        return P

    def solve(self, problem: Dict[str, List[List[int]]]) -> Dict[str, List[float]]:
        """Compute PageRank vector by power iteration."""
        adj_list = problem.get("adjacency_list", [])
        n = len(adj_list)

        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}

        P = self._build_matrices(adj_list)

        # Teleportation vector
        teleport = np.full(n, 1.0 / n, dtype=np.float64)

        # Initial rank vector (uniform)
        r = np.full(n, 1.0 / n, dtype=np.float64)

        alpha = self.alpha
        tol = self.tol
        max_iter = self.max_iter

        for _ in range(max_iter):
            r_new = alpha * P @ r + (1.0 - alpha) * teleport
            diff = np.abs(r_new - r).sum()
            if diff < tol:
                r = r_new
                break
            r = r_new

        # Normalise to avoid any drift
        r /= r.sum()
        return {"pagerank_scores": r.tolist()}
