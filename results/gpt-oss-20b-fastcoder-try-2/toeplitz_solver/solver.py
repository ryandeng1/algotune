from typing import Any
import numpy as np
from scipy.linalg import solve_toeplitz

class Solver:
    def solve(self, problem: dict[str, list[float]]) -> list[float]:
        """
        Solve the linear system Tx = b using scipy's efficient Toeplitz solver.

        The inputs are already in list form; convert them to numpy arrays without copying.
        """
        # avoid unnecessary copies – the inputs are already 1‑D
        c = np.asarray(problem["c"], dtype=np.float64, order="C")
        r = np.asarray(problem["r"], dtype=np.float64, order="C")
        b = np.asarray(problem["b"], dtype=np.float64, order="C")
        x = solve_toeplitz((c, r), b)
        return x.tolist()