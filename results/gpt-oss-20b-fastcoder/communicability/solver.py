from typing import Any, Dict, List
import numpy as np
from scipy.linalg import expm

class Solver:
    """
    Optimised communicator calculation using fast NumPy/SciPy routines.
    """

    def solve(self, problem: Dict[str, List[List[int]]]) -> Dict[str, Dict[int, Dict[int, float]]]:
        """
        Calculates the communicability for an undirected graph.

        Parameters
        ----------
        problem : dict
            Dictionary containing the adjacency list of the graph under the key
            ``'adjacency_list'``.  The list must be zero based and the graph
            is assumed to be undirected.

        Returns
        -------
        dict
            Dictionary containing the communicability matrix as a dictionary of
            dictionaries:
                {"communicability": {u: {v: value, ...}, ...}}
        """
        adj_list = problem.get("adjacency_list", [])
        n = len(adj_list)
        if n == 0:
            return {"communicability": {}}

        # Build adjacency matrix – use bool to reduce memory  
        A = np.zeros((n, n), dtype=np.float64)
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                if 0 <= v < n:      # guard against bad indices
                    A[u, v] = 1.0
                    A[v, u] = 1.0    # symmetrise

        # Communicability = exp(A)
        C = expm(A)

        # Convert to dict of dicts (float)
        result: Dict[int, Dict[int, float]] = {}
        for i in range(n):
            row = C[i]
            dic = {j: float(row[j]) for j in range(n)}
            result[i] = dic

        return {"communicability": result}