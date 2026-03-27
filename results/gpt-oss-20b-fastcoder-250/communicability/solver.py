import numpy as np
from scipy.linalg import expm

def solve(problem: dict[str, list[list[int]]]) -> dict[str, dict[int, dict[int, float]]]:
    """
    Calculate the communicability matrix of an unweighted undirected graph.
    Communicability is defined as the exponential of the adjacency matrix.
    The resulting matrix is returned as a nested dictionary:
        {u: {v: value}}

    Parameters
    ----------
    problem
        Dictionary with key 'adjacency_list' mapping to a list of neighbor
        indices for each node (0‑based).

    Returns
    -------
    dict
        {"communicability": [{u: {v: float}}]} – a dictionary of dictionaries
        containing the communicability values.
    """
    adj_list = problem.get("adjacency_list", [])
    n = len(adj_list)
    if n == 0:
        return {"communicability": {}}

    # Build the adjacency matrix once, without NetworkX
    A = np.zeros((n, n), dtype=np.float64)
    for u, neighbors in enumerate(adj_list):
        for v in neighbors:
            if 0 <= v < n:
                A[u, v] = 1.0
                A[v, u] = 1.0  # undirected

    # Compute the communicability matrix: exp(A)
    C = expm(A)

    # Convert to nested dictionary
    comm_dict: dict[int, dict[int, float]] = {
        u: {v: float(C[u, v]) for v in range(n)} for u in range(n)
    }
    return {"communicability": comm_dict}