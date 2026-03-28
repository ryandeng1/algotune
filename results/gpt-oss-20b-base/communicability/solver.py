from typing import Dict, List
import numpy as np
from scipy.linalg import expm

def solve(problem: Dict[str, List[List[int]]]) -> Dict[str, Dict[int, Dict[int, float]]]:
    """
    Compute the communicability matrix for a graph defined by an adjacency list.
    The communicability between nodes u and v is defined as the (u,v) entry of the
    matrix exponential of the adjacency matrix.

    The result is a nested dictionary:
        { u: { v: communability[u][v] } }

    This implementation avoids the overhead of NetworkX by working directly with
    NumPy / SciPy. It is compatible with Python 3.10.

    Args:
        problem: Dictionary containing the adjacency list under the key
                 "adjacency_list". The graph is undirected and unweighted.

    Returns:
        Dictionary with a single key "communicability" whose value is the
        communicability matrix in dictionary form.
    """
    adj_list = problem.get("adjacency_list", [])
    n = len(adj_list)
    if n == 0:
        return {"communicability": {}}

    # Build the adjacency matrix
    A = np.zeros((n, n), dtype=float)
    for u, neighbors in enumerate(adj_list):
        for v in neighbors:
            if 0 <= v < n:
                A[u, v] = 1.0
                A[v, u] = 1.0  # undirected

    # Compute the matrix exponential
    expA = expm(A)

    # Convert matrix to dict of dicts
    result = {
        u: {v: float(expA[u, v]) for v in range(n)} for u in range(n)
    }

    return {"communicability": result}