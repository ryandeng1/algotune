import numpy as np
from scipy.linalg import solve_toeplitz

class Solver:
    def solve(self, problem: dict[str, list[float]]) -> list[float]:
        """Solve Tx = b using scipy's Levinson-Durbin implementation."""
        # Use np.asarray to avoid unnecessary copies if inputs already arrays
        c = np.asarray(problem['c'])
        r = np.asarray(problem['r'])
        b = np.asarray(problem['b'])
        x = solve_toeplitz((c, r), b)
        return x.tolist()