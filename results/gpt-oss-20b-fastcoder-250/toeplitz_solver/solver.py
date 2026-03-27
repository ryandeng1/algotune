import numpy as np
from scipy.linalg import solve_toeplitz

class Solver:
    def solve(self, problem: dict[str, list[float]]) -> list[float]:
        """
        Solve the linear system Tx = b using the efficient Toeplitz solver from SciPy.
        The input data is converted to NumPy arrays in a single step to avoid
        unnecessary intermediate copies, and the output is returned as a plain list.
        """
        c = np.asarray(problem["c"], dtype=np.float64)
        r = np.asarray(problem["r"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        x = solve_toeplitz((c, r), b)
        return x.tolist()