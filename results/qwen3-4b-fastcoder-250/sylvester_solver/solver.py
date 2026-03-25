import numpy as np
from scipy.linalg import solve_sylvester

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A = problem["A"]
        B = problem["B"]
        Q = problem["Q"]
        X = solve_sylvester(A, B, Q, check_finite=False)
        return {"X": X}
