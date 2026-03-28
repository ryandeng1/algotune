import numpy as np
from scipy.linalg import solve_toeplitz

class Solver:
    """
    Fast Toeplitz system solver.
    """
    def solve(self, problem: dict[str, list[float]]) -> list[float]:
        """
        Solve Tx = b using the Toeplitz solver from SciPy.

        Parameters
        ----------
        problem : dict[str, list[float]]
            A dictionary containing the Toeplitz system components:
            - 'c': first column of T (as list)
            - 'r': first row  of T (as list)
            - 'b': right‑hand side vector (as list)

        Returns
        -------
        list[float]
            The solution vector x.
        """
        # Convert lists to 1‑D NumPy arrays (no copy if already)
        c = np.asarray(problem['c'], dtype=float, order='C')
        r = np.asarray(problem['r'], dtype=float, order='C')
        b = np.asarray(problem['b'], dtype=float, order='C')

        # Solve (T) x = b
        x = solve_toeplitz((c, r), b)

        # Return python list for API compatibility
        return x.tolist()