# solver.py
import numpy as np
from numba import njit, types
from numba.typed import List

@njit(types.List(types.Array(types.float64, 1, "C")))
def _project_to_simplex(y):
    """
    Project y onto the probability simplex:
    min ||x - y||_2^2   s.t. 1^T x = 1, x_i >= 0
    """
    # Sort y in descending order
    sorted_y = np.sort(y)[::-1]
    n = sorted_y.shape[0]
    # Cumulative sum minus 1
    cumsum_y = np.cumsum(sorted_y) - 1.0

    # Compute the threshold index rho
    # We check condition sorted_y > cumsum_y / (1:n)
    # Using a while loop (numba-friendly)
    rhs = cumsum_y / np.arange(1, n + 1)
    mask = sorted_y > rhs

    # Find last index where mask is True
    # Equivalent to np.where(mask)[0][-1]
    rho = -1
    for i in range(n):
        if mask[i]:
            rho = i
    # At least one element will satisfy the condition
    theta = cumsum_y[rho] / (rho + 1)

    # Compute final projection
    x = np.empty_like(y)
    for i in range(n):
        val = y[i] - theta
        x[i] = val if val > 0 else 0.0
    return List.from_array(x, dtype=types.float64)

class Solver:
    def solve(self, problem):
        """
        Solve the quadratic programming problem (Euclidean projection onto the probability simplex)
        in O(n log n) time using a numba-accelerated routine.
        """
        y = np.asarray(problem.get('y'), dtype=np.float64)
        if y.ndim > 1:
            y = y.ravel()
        # Use the numba implementation
        proj = _project_to_simplex(y)
        # Convert back to a numpy array for consistency
        return {'solution': np.asarray(proj, dtype=np.float64)}