import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray[np.float64]) -> NDArray[np.float64]:
        n = problem.shape[0]
        i = np.arange(n)
        j = np.arange(n)
        i_grid, j_grid = np.meshgrid(i, j)
        arg = np.pi * (2 * i_grid + 1) * (j_grid + 1) / (2 * n)
        S = np.sin(arg)
        
        T1 = np.dot(S, problem.T).T
        T2 = np.dot(S, T1.T).T
        return T2
