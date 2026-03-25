# solver.py
import numpy as np
from scipy.linalg import solve_sylvester

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the Sylvester equation AX + XB = Q for X.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - A : (n, n) complex numpy array
                - B : (m, m) complex numpy array
                - Q : (n, m) complex numpy array
                - random_seed : int (ignored by this solver)

        Returns
        -------
        dict
            Dictionary with a single key 'X' containing the solution matrix X.
        """
        A = problem["A"]
        B = problem["B"]
        Q = problem["Q"]

        # Use the highly optimized LAPACK routine wrapped by scipy
        X = solve_sylvester(A, B, Q)

        return {"X": X}
