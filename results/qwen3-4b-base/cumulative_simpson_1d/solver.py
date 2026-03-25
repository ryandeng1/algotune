import numpy as np
from numba import njit

class Solver:
    def solve(self, problem: dict) -> np.ndarray:
        y = problem["y"]
        dx = problem["dx"]
        n = len(y)
        result = np.zeros(n)
        
        @njit
        def compute():
            for i in range(1, n):
                if i % 2 == 0:
                    result[i] = result[i-1] + dx * (y[i-2] + 4*y[i-1] + y[i]) / 3.0
                else:
                    result[i] = result[i-1]
        
        compute()
        return result
