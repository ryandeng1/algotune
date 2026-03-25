import numpy as np
from numba import njit

@njit
def solve_toeplitz_fast(c, r, b):
    n = len(b)
    x = np.zeros(n, dtype=np.float64)
    for k in range(n):
        if k == 0:
            x[k] = b[k] / c[k]
        else:
            total = 0.0
            for j in range(k):
                total += c[k - j] * x[j]
            x[k] = (b[k] - total) / r[0]
    return x

class Solver:
    def solve(self, problem: dict[str, list[float]]) -> list[float]:
        c = np.array(problem["c"])
        r = np.array(problem["r"])
        b = np.array(problem["b"])
        return solve_toeplitz_fast(c, r, b).tolist()
