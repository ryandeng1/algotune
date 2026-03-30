# solver.py
import numpy as np
from scipy.linalg import solve_sylvester
from typing import Any, Dict

class Solver:
    """Fast solver for the Sylvester equation X*A + B*X = Q."""

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parameters
        ----------
        problem : dict
            Dictionary containing three numpy arrays:
            - 'A': (n, n) array
            - 'B': (m, m) array
            - 'Q': (m, n) array

        Returns
        -------
        dict
            Dictionary with the solution array under key 'X'.
        """
        # Fetch arrays once from the dict to reduce lookup overhead.
        A = problem['A']
        B = problem['B']
        Q = problem['Q']

        # Solve the Sylvester equation.  solve_sylvester is a thin wrapper
        # around LAPACK's *trsm* routine and is already highly optimised
        # for speed.
        X = solve_sylvester(A, B, Q)

        return {'X': X}