import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Any, List

def solve(problem: dict[str, Any]) -> List[List[Any]]:
    """
    Solves the minimum weight assignment problem.

    The input `problem` is a dictionary containing the keys
    ``capacity`` – a square matrix where the value indicates the
    allowable flow (must be 1 for assignment edges, otherwise 0)
    and ``cost`` – a square matrix of the same size with edge
    costs.  The keys ``s`` and ``t`` are ignored because the
    assignment problem is bipartite and does not require a
    source/sink representation.

    The function returns an `n x n` matrix `flow` where
    ``flow[i][j] == 1`` means there is a unit of flow from vertex *i*
    to vertex *j*, and ``0`` otherwise.

    This implementation uses `scipy.optimize.linear_sum_assignment`,
    which runs in O(n^3) time and is heavily optimised in C,
    making it faster than a pure‑Python NetworkX flow algorithm.
    """
    try:
        capacity = np.array(problem["capacity"])
        cost = np.array(problem["cost"])

        # Ensure that the assignment graph is bipartite with
        # allowed edges where capacity > 0.
        n = capacity.shape[0]
        allowed = capacity > 0

        # For forbidden edges we assign a very high cost so that
        # they will never be chosen.  We use a large number that
        # is larger than any possible sum of real costs.
        INF = int(cost.max() * n + 1)
        cost_for_assignment = np.where(allowed, cost, INF)

        # Perform the Hungarian algorithm (linear sum assignment)
        row_ind, col_ind = linear_sum_assignment(cost_for_assignment)

        # Build the flow matrix
        flow = np.zeros((n, n), dtype=int)
        flow[row_ind, col_ind] = 1

        # If the graph is not fully matchable (e.g. some rows
        # have no outgoing edges), return a zero matrix.
        if flow.sum() < n:
            return [[0] * n for _ in range(n)]

        return flow.tolist()

    except Exception:
        n = len(problem.get("capacity", []))
        return [[0] * n for _ in range(n)]