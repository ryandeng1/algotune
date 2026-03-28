from typing import Any
import numpy as np
import scipy.linalg

class Solver:
    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, dict[int, dict[int, float]]]:
        """
        Compute the communicability matrix of an undirected simple graph.

        Communicability is defined as the matrix exponential of the adjacency
        matrix.  All operations are performed in NumPy / SciPy, which is much
        faster than the NetworkX implementation used in the original code.

        Args:
            problem: {"adjacency_list": adj_list}
                adjacency list is a list of lists; node indices are 0..n-1.
        Returns:
            {"communicability": comm_dict}
                comm_dict[u][v] is the communicability between u and v.
        """
        adj_list = problem.get('adjacency_list', [])
        n = len(adj_list)
        if n == 0:
            return {"communicability": {}}

        # Build adjacency matrix
        A = np.zeros((n, n), dtype=np.float64)
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                # undirected: ensure symmetric
                A[u, v] = 1.0
                A[v, u] = 1.0

        try:
            # communicability ≈ exp(adj_matrix)
            # Use dense matrix exponential; it's fast for moderate n
            ExpA = scipy.linalg.expm(A)
        except Exception:
            return {"communicability": {}}

        # Convert to required dict-of-dicts structure
        result_comm_dict: dict[int, dict[int, float]] = {}
        for u in range(n):
            row = ExpA[u]
            result_comm_dict[u] = {v: float(row[v]) for v in range(n)}

        return {"communicability": result_comm_dict}