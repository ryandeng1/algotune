import numpy as np
from numba import njit

@njit
def dct1d(x):
    N = x.shape[0]
    result = np.zeros(N, dtype=np.float64)
    for k in range(N):
        for m in range(N):
            result[k] += x[m] * np.cos(np.pi * (2 * m + 1) * k / (2 * N))
    return result

class Solver:
    def solve(self, problem: np.ndarray) -> np.ndarray:
        M = problem.shape[0]
        rows = np.zeros_like(problem)
        for i in range(M):
            rows[i] = dct1d(problem[i])
        result = np.zeros_like(rows)
        for j in range(M):
            result[:, j] = dct1d(rows[:, j])
        return result
