# solver.py
import math
from typing import Any, Dict, List

import numpy as np


class Solver:
    """
    Efficient PageRank computation using NumPy power iteration.
    """

    # Default damping factor and convergence parameters
    alpha: float = 0.85
    tol: float = 1e-12
    max_iter: int = 100

    def _build_matrix(self, adj_list: List[List[int]]) -> np.ndarray:
        """
        Construct the column‑stochastic transition matrix P.
        """
        n = len(adj_list)
        P = np.zeros((n, n), dtype=float)

        for i, neighbors in enumerate(adj_list):
            out_deg = len(neighbors)
            if out_deg == 0:
                # Dangling node: link equally to all nodes
                P[:, i] = 1.0 / n
            else:
                share = 1.0 / out_deg
                P[neighbors, i] = share
        return P

    def solve(self, problem: Dict[str, List[List[int]]]) -> Dict[str, List[float]]:
        """
        Compute PageRank scores for a directed graph.

        Parameters
        ----------
        problem : dict
            {"adjacency_list": adj_list}
        Returns
        -------
        dict
            {"pagerank_scores": [score0, score1, ...]}
        """
        adj_list = problem["adjacency_list"]
        n = len(adj_list)

        # Handle trivial cases
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}

        # Build transition matrix
        P = self._build_matrix(adj_list)

        # Teleportation vector
        teleport = np.full(n, 1.0 / n, dtype=float)

        # Initial PageRank vector (uniform)
        r = np.full(n, 1.0 / n, dtype=float)

        # Power iteration
        for _ in range(self.max_iter):
            r_new = self.alpha * P @ r + (1.0 - self.alpha) * teleport
            # L1 normalization
            diff = np.abs(r_new - r).sum()
            r = r_new
            if diff < self.tol:
                break

        # Ensure scores are finite
        r = np.where(np.isfinite(r), r, 0.0)

        # Final normalisation
        total = r.sum()
        if total > 0:
            r /= total

        return {"pagerank_scores": r.tolist()}
