import numpy as np
from scipy.linalg import expm
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, Dict[int, Dict[int, float]]]:
        """
        Computes the communicability matrix as exp(A) where A is the adjacency matrix.
        This implementation uses NumPy and SciPy for efficient matrix exponentiation
        and avoids the overhead of NetworkX.
        """
        adj_list: List[List[int]] = problem.get("adjacency_list", [])
        n = len(adj_list)

        if n == 0:
            return {"communicability": {}}

        # Build adjacency matrix
        A = np.zeros((n, n), dtype=float)
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                if u <= v:
                    A[u, v] = 1.0
                    if u != v:
                        A[v, u] = 1.0

        # Compute communicability matrix: exp(A)
        try:
            comm_matrix = expm(A)
        except Exception:
            # Fallback to empty result on failure
            return {"communicability": {}}

        # Convert to dict of dicts with float values
        result: Dict[int, Dict[int, float]] = {}
        for i in range(n):
            row: Dict[int, float] = {}
            for j in range(n):
                row[j] = float(comm_matrix[i, j])
            result[i] = row

        return {"communicability": result}