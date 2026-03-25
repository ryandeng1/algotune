# solver.py
import numpy as np
from scipy.linalg import solve_toeplitz

class Solver:
    def solve(self, problem: dict[str, list[float]]) -> list[float]:
        """
        Solve the linear system Tx = b where T is a Toeplitz matrix.
        Parameters
        ----------
        problem : dict
            Dictionary with keys 'c', 'r', and 'b' representing the first column,
            first row of the Toeplitz matrix and the right-hand side vector respectively.

        Returns
        -------
        list[float]
            Solution vector x as a plain Python list.
        """
        # Convert input lists to NumPy arrays for efficient computation
        c = np.asarray(problem["c"], dtype=np.float64)
        r = np.asarray(problem["r"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        # Solve the Toeplitz system using the fast Levinson-Durbin algorithm
        x = solve_toeplitz((c, r), b)

        # Convert the result to a Python list before returning
        return x.tolist()
