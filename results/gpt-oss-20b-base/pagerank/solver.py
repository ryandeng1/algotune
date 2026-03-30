# solver.py
import numpy as np
from typing import Dict, List


class Solver:
    def __init__(self) -> None:
        # Teleportation probability
        self.alpha = 0.85
        self.max_iter = 100
        self.tol = 1e-06

    def _pagerank(self, adj: List[List[int]]) -> List[float]:
        """
        Compute PageRank using a vectorised power‑iteration algorithm.
        The input adjacency list is converted to a NumPy 0/1 matrix.
        """
        n = len(adj)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        # Build adjacency matrix (boolean for speed)
        data = np.zeros((n, n), dtype=np.uint8)
        for src, dsts in enumerate(adj):
            if dsts:
                data[src, dsts] = 1

        # Row‑normalise so each row sums to 1; dangling rows -> all zeros
        row_sums = data.sum(axis=1, keepdims=True)
        dangling = row_sums.squeeze() == 0
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        M = data / row_sums

        # Power iteration
        rank = np.full(n, 1.0 / n, dtype=np.float64)
        click = np.zeros(n, dtype=np.float64)

        for _ in range(self.max_iter):
            # Contribution from links
            click[:] = M.T @ rank

            # Handle dangling nodes: distribute uniformly
            if dangling.any():
                click += rank[dangling].sum() / n

            # Teleportation (damping)
            rank[:] = self.alpha * click + (1.0 - self.alpha) / n

            # Convergence check
            if np.linalg.norm(rank - click, ord=1) < self.tol:
                break

        return rank.tolist()

    def solve(self, problem: Dict[str, List[List[int]]]) -> Dict[str, List[float]]:
        """
        Calculates the PageRank scores for the graph.

        Parameters
        ----------
        problem : dict
            {"adjacency_list": adj_list}

        Returns
        -------
        dict
            {"pagerank_scores": [score_node_0, score_node_1, ...]}
        """
        adj_list = problem.get("adjacency_list", [])
        scores = self._pagerank(adj_list)
        return {"pagerank_scores": scores}