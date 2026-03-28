import numpy as np
from scipy.linalg import solve_sylvester

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A, B, Q = problem["A"], problem["B"], problem["Q"]
        # Ensure inputs are NumPy arrays for maximum performance
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        Q = np.asarray(Q, dtype=np.float64)
        X = solve_sylvester(A, B, Q)
        return {"X": X}