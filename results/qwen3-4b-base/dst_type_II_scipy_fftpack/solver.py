import numpy as np
from numba import njit

@njit
def _dst1d(x):
    n = x.shape[0]
    res = np.zeros(n, dtype=x.dtype)
    for k in range(n):
        for m in range(n):
            res[k] += x[m] * np.sin((m + 0.5) * (k + 0.5) * np.pi / n)
    return res

class Solver:
    def solve(self, problem: np.ndarray) -> np.ndarray:
        n = problem.shape[0]
        rows = np.zeros_like(problem)
        for i in range(n):
            rows[i] = _dst1d(problem[i])
        result = np.zeros_like(rows)
        for j in range(n):
            col = rows[:, j]
            result[:, j] = _dst1d(col)
        return result
