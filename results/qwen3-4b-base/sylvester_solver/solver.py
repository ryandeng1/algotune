from typing import Any
import numpy as np
from scipy.linalg import solve_sylvester

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A, B, Q = problem["A"], problem["B"], problem["Q"]
        X = solve_sylvester(A, B, Q, check_finite=False)
        return {"X": X}
