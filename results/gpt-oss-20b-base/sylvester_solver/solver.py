# solver.py
import numpy as np
from scipy.linalg import solve_sylvester

class Solver:
    """
    Solver for the Sylvester matrix equation AX + X B = Q.
    """

    def solve(self, problem) -> dict[str, np.ndarray]:
        """
        Solves AX + X B = Q for X.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                A : array_like, shape (n, n)
                B : array_like, shape (m, m)
                Q : array_like, shape (n, m)

        Returns
        -------
        dict
            Dictionary containing the solution matrix X.
        """
        A = np.asarray(problem["A"], dtype=float, copy=False)
        B = np.asarray(problem["B"], dtype=float, copy=False)
        Q = np.asarray(problem["Q"], dtype=float, copy=False)

        # The underlying C implementation is already highly optimised.
        X = solve_sylvester(A, B, Q)

        return {"X": X}