from typing import Any
import numpy as np
from scipy.linalg import solve_toeplitz


class Solver:
    def solve(self, problem: dict[str, list[float]]) -> list[float]:
        c = np.array(problem["c"], dtype=np.float64)
        r = np.array(problem["r"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        x = solve_toeplitz((c, r), b)
        return x.tolist()