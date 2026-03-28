import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, List[List[int]]]) -> Dict[str, Dict[int, Dict[int, float]]]:
        """
        Calculates the communicability matrix of an undirected graph
        defined by an adjacency list.  Communicability is defined as
        the matrix exponential of the unweighted adjacency matrix
        (summing all walks weighted by 1/k!).

        Parameters
        ----------
        problem : dict
            {"adjacency_list": adj_list}
            adj_list is a list of lists, where adj_list[u] contains
            the neighbors of vertex u.

        Returns
        -------
        dict
            {"communicability": comm_dict}
            comm_dict[u][v] is the communicability between nodes u and v.
        """
        adj_list = problem.get("adjacency_list", [])
        n = len(adj_list)
        if n == 0:
            return {"communicability": {}}

        # Build symmetric adjacency matrix
        A = np.zeros((n, n), dtype=float)
        for u, neigh in enumerate(adj_list):
            for v in neigh:
                A[u, v] = 1.0
                A[v, u] = 1.0

        # Eigen‑decomposition of a symmetric matrix
        w, Q = np.linalg.eigh(A)          # w: eigenvalues, Q: orthonormal eigenvectors
        exp_w = np.exp(w)                 # e^(eigenvalue)

        # Construct exp(A) = Q * exp(D) * Q^T
        # Using elementwise multiplication of columns of Q by exp_w
        exp_A = (Q * exp_w) @ Q.T

        # Convert to nested dict of floats
        comm_dict: Dict[int, Dict[int, float]] = {}
        for i in range(n):
            row = {}
            ai = exp_A[i]
            for j in range(n):
                row[j] = float(ai[j])
            comm_dict[i] = row

        return {"communicability": comm_dict}