# solver.py
import math
from typing import Any, Dict
import numpy as np
from scipy.linalg import expm

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Dict[int, Dict[int, float]]]:
        """
        Compute the communicability matrix of an undirected graph
        given by its adjacency list.  The communicability is defined
        as the matrix exponential of the adjacency matrix.
        The result is returned as a nested dict:
            {u: {v: value, ...}, ...}
        where u,v are node indices.
        """
        # Grab adjacency list
        adj_list = problem.get("adjacency_list", [])
        n = len(adj_list)

        # Empty graph – return empty dict
        if n == 0:
            return {"communicability": {}}

        # Build adjacency matrix as a NumPy array
        A = np.zeros((n, n), dtype=np.float64)
        for i, neighbors in enumerate(adj_list):
            for j in neighbors:
                # Adjacency is symmetric
                A[i, j] = 1.0

        # Compute matrix exponential once
        expA = expm(A)

        # Convert to nested dict structure
        comm: Dict[int, Dict[int, float]] = {}
        for i in range(n):
            row_dict: Dict[int, float] = {}
            row = expA[i]
            for j in range(n):
                row_dict[j] = float(row[j])
            comm[i] = row_dict

        return {"communicability": comm}
