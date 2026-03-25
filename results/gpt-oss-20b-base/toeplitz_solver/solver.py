import numpy as np
from scipy.linalg import solve_toeplitz

class Solver:
    def solve(self, problem: dict[str, list[float]]) -> list[float]:
        # Convert inputs to NumPy arrays (fast, memory‑efficient)
        c = np.asarray(problem["c"], dtype=np.float64)
        r = np.asarray(problem["r"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        # Use the efficient Levinson–Durbin solver from SciPy
        x = solve_toeplitz((c, r), b)

        # Return a plain Python list for the harness
        return x.tolist()
