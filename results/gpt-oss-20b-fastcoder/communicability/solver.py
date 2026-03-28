from typing import Any, Dict, List
import numpy as np
from scipy.linalg import expm

class Solver:
    def solve(self, problem: Dict[str, List[List[int]]]) -> Dict[str, Dict[int, Dict[int, float]]]:
        """
        Calculates the communicability for the graph using the matrix exponential.
        Communicability C_{uv} = (exp(A))_{uv}, where A is the adjacency matrix.
        The graph is undirected and unweighted.
        """
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        if n == 0:
            return {"communicability": {}}

        # Build adjacency matrix
        A = np.zeros((n, n), dtype=float)
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                if u <= v:          # undirected graph; avoid duplicate edges
                    A[u, v] = 1.0
                    A[v, u] = 1.0

        # Compute matrix exponential once
        try:
            expA = expm(A)
        except Exception:
            return {"communicability": {}}

        # Convert to dictionary of dictionaries
        comm_dict: Dict[int, Dict[int, float]] = {}
        for i in range(n):
            row = expA[i]
            comm_dict[i] = {j: float(row[j]) for j in range(n)}

        return {"communicability": comm_dict}