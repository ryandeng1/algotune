from typing import Any, List
import numpy as np
from scipy.optimize import linear_sum_assignment

def solve(problem: dict[str, Any]) -> List[List[Any]]:
    """
    Solves a minimum cost perfect assignment (Hungarian algorithm).
    The problem dict must contain:
        - 'cost': a square 2‑D list or array of costs
        - 'capacity': a square 2‑D list or array; positions with capacity 0 are prohibited
    Returns a matrix `flow` where flow[i][j] is 1 if row i is assigned to column j,
    and 0 otherwise.
    """
    # Extract data and convert to numpy arrays
    cost = np.array(problem["cost"], dtype=float)
    capacity = np.array(problem["capacity"], dtype=int)

    # Ensure square matrices
    if cost.shape != capacity.shape:
        raise ValueError("cost and capacity matrices must have the same shape")
    n = cost.shape[0]

    # Positions with no capacity are made infeasible (very large cost)
    LARGE = 1e12
    mask = capacity == 0
    cost_for_algo = cost.copy()
    cost_for_algo[mask] = LARGE

    # Apply Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_for_algo)

    # Build solution matrix
    solution = np.zeros((n, n), dtype=int)
    for r, c in zip(row_ind, col_ind):
        # Only assign if the original capacity allows it
        if capacity[r, c] > 0:
            solution[r, c] = 1

    # Convert to plain Python list of lists
    return solution.tolist()