import numpy as np
from scipy.linalg import solve_toeplitz

class Solver:
    def solve(self, problem: dict[str, list[float]]) -> list[float]:
        """Solve the linear system Tx = b using scipy's Toeplitz solver."""
        # Convert to NumPy arrays with the most efficient dtype
        c = np.array(problem["c"], dtype=np.float64, copy=False)
        r = np.array(problem["r"], dtype=np.float64, copy=False)
        b = np.array(problem["b"], dtype=np.float64, copy=False)

        # Directly solve without any intermediate copies
        x = solve_toeplitz((c, r), b, overwrite_b=True, overwrite_c=True, overwrite_r=True)

        return x.tolist()