import numpy as np
from typing import Any, Dict, List

# Try to use scipy for a fast matrix exponential, otherwise fall back
try:
    from scipy.linalg import expm
except Exception:
    def expm(A: np.ndarray) -> np.ndarray:
        """
        Very rough matrix exponential via Taylor series (good for small n).
        """
        n = A.shape[0]
        result = np.eye(n, dtype=float)
        term = np.eye(n, dtype=float)
        k = 1
        while True:
            term = term @ A / k
            new = result + term
            if np.linalg.norm(new - result, ord='fro') < 1e-9:
                return new
            result = new
            k += 1

def solve(problem: Dict[str, List[List[int]]]) -> Dict[str, Any]:
    """
    Compute the communicability matrix for an undirected graph represented by an
    adjacency list.
    """
    adj_list = problem.get("adjacency_list", [])
    n = len(adj_list)
    if n == 0:
        return {"communicability": {}}

    # Construct adjacency matrix
    A = np.zeros((n, n), dtype=float)
    for i, neighbors in enumerate(adj_list):
        for j in neighbors:
            if 0 <= j < n:
                A[i, j] = 1.0
    # Make it symmetric (undirected)
    A = np.triu(A) + np.triu(A, 1).T

    # Matrix exponential gives communicability
    C = expm(A)

    # Convert to the required dict-of-dicts format
    comm_dict: Dict[int, Dict[int, float]] = {}
    for i in range(n):
        row = {}
        for j in range(n):
            row[j] = float(C[i, j])
        comm_dict[i] = row

    return {"communicability": comm_dict}