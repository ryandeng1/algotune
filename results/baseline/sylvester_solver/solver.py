from typing import Any
from scipy.linalg import solve_sylvester

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A, B, Q = (problem['A'], problem['B'], problem['Q'])
        X = solve_sylvester(A, B, Q)
        return {'X': X}
