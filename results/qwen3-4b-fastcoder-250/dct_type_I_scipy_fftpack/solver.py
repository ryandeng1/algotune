import numpy as np

class Solver:
    def solve(self, problem: np.ndarray) -> np.ndarray:
        N = problem.shape[0]
        m = np.arange(N)
        k = np.arange(N)
        angles = np.pi * (m + 0.5) * k / (2 * N)
        cosines = np.cos(angles)
        rows = np.dot(problem, cosines.T)
        cols = np.dot(rows.T, cosines)
        return cols.T
