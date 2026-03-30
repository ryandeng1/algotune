# solver.py
import numpy as np
from typing import Dict, List

class Solver:
    """Fast PageRank solver using NumPy power iteration."""
    __slots__ = ("damping", "max_iter", "tol")

    def __init__(self) -> None:
        self.damping: float = 0.85
        self.max_iter: int = 100
        self.tol: float = 1e-6

    def solve(self, problem: Dict[str, List[List[int]]]) -> Dict[str, List[float]]:
        """Return PageRank scores for the directed graph in `problem`."""
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}

        # Build transition matrix P: P[j, i] is probability from i to j
        # Use sparse logic with NumPy arrays for speed
        row, col, data = [], [], []
        for src, nbrs in enumerate(adj_list):
            if not nbrs:  # dangling node, will be handled separately
                continue
            prob = 1.0 / len(nbrs)
            row.extend(nbrs)
            col.extend([src] * len(nbrs))
            data.extend([prob] * len(nbrs))

        P = np.zeros((n, n), dtype=np.float64)
        if row:
            P[np.array(row), np.array(col)] = np.array(data)

        # Identify dangling nodes
        outdeg = np.array([len(nbrs) for nbrs in adj_list], dtype=np.int64)
        dangling = (outdeg == 0)

        # Precompute constants
        teleport = (1.0 - self.damping) / n
        vector = np.full(n, 1.0 / n, dtype=np.float64)

        for _ in range(self.max_iter):
            # Power iteration: vector = damping * P @ vector + teleport
            new_vector = P @ vector
            # Add dangling contribution
            if dangling.any():
                new_vector += vector[dangling].sum()
            new_vector = self.damping * new_vector + teleport
            # Check convergence
            if np.linalg.norm(new_vector - vector, 1) < self.tol:
                vector = new_vector
                break
            vector = new_vector

        return {"pagerank_scores": vector.tolist()}