from __future__ import annotations
from typing import Dict, List

# Constants controlling the series truncation
_MAX_K = 25            # maximum number of terms for the series
_TOL = 1e-12           # threshold for negligible terms

def _mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Multiply two square matrices A and B.  Assumes both are n x n."""
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for k in range(n):
            aik = Ai[k]
            if aik:
                Bk = B[k]
                for j in range(n):
                    Ci[j] += aik * Bk[j]
    return C

def _mat_add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Add two square matrices A and B."""
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Ai, Bi, Ci_row = A[i], B[i], C[i]
        for j in range(n):
            Ci_row[j] = Ai[j] + Bi[j]
    return C

def _mat_scale(A: List[List[float]], scalar: float) -> List[List[float]]:
    """Scale matrix A by scalar."""
    n = len(A)
    return [[scalar * A[i][j] for j in range(n)] for i in range(n)]

class Solver:
    def solve(self, problem: Dict[str, List[List[int]]]) -> Dict[str, Dict[int, Dict[int, float]]]:
        """
        Compute the communicability matrix for an undirected graph.
        Communicability is defined as (e^A) where A is the adjacency matrix.
        The result is returned as a nested dictionary: {u: {v: value}}.
        """
        adj_list = problem.get("adjacency_list", [])
        n = len(adj_list)
        if n == 0:
            return {"communicability": {}}

        # Build adjacency matrix
        A = [[0.0] * n for _ in range(n)]
        for u, nbrs in enumerate(adj_list):
            for v in nbrs:
                A[u][v] = 1.0

        # Initialize result with identity (k=0 term)
        result = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        term = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]  # A^0 / 0!

        # Compute series sum k=1.._MAX_K
        for k in range(1, _MAX_K + 1):
            term = _mat_mul(term, A)          # term = A^{k}
            inv_fact = 1.0 / (k * (k - 1))
            scaled = _mat_scale(term, inv_fact)  # term / k!
            result = _mat_add(result, scaled)
            # Termination check based on maximum element
            max_elem = max(max(row) for row in scaled)
            if max_elem < _TOL:
                break

        # Convert matrix to dict-of-dicts
        commun_dict: Dict[int, Dict[int, float]] = {}
        for i in range(n):
            row_dict = {}
            for j in range(n):
                row_dict[j] = float(result[i][j])
            commun_dict[i] = row_dict

        return {"communicability": commun_dict}