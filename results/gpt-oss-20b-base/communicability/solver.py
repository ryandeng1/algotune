from typing import Any, Dict, List
import numpy as np
from scipy.linalg import expm


class Solver:
    """
    Computes the communicability matrix of an undirected graph.
    For a graph with adjacency matrix `A`, communability equals `exp(A)`.
    """

    def solve(self, problem: Dict[str, List[List[int]]]) -> Dict[str, Dict[int, Dict[int, float]]]:
        adj_list = problem.get("adjacency_list", [])
        n = len(adj_list)
        if n == 0:
            return {"communicability": {}}

        # Build adjacency matrix
        A = np.zeros((n, n), dtype=np.float64)
        for u, neigh in enumerate(adj_list):
            for v in neigh:
                if 0 <= v < n:
                    A[u, v] = 1.0
        # symmetrize (undirected graph)
        A = np.maximum(A, A.T)

        # Compute exp(A)
        try:
            expA = expm(A)
        except Exception:
            return {"communicability": {}}

        # Convert to nested dict
        comm = {
            u: {v: float(expA[u, v]) for v in range(n)}
            for u in range(n)
        }

        return {"communicability": comm}