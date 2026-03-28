from typing import Any
import numpy as np
from scipy.linalg import solve_toeplitz

class Solver:
    def solve(self, problem: dict[str, list[float]]) -> list[float]:
        # Convert lists to NumPy arrays in one step and solve with Levinson-Durbin
        c, r, b = (np.asarray(problem[k], dtype=float) for k in ('c', 'r', 'b'))
        return solve_toeplitz((c, r), b).tolist()