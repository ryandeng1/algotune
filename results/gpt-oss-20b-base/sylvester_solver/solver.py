from typing import Any
import numpy as np
from scipy.linalg import solve_sylvester

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Solves the Sylvester equation AX + X B = Q for X.

        Parameters
        ----------
        problem : dict
            Dictionary containing numpy arrays 'A', 'B', and 'Q'.
        
        Returns
        -------
        dict
            Dictionary with key 'X' containing the solution matrix.
        """
        A = problem["A"]
        B = problem["B"]
        Q = problem["Q"]

        X = solve_sylvester(A, B, Q)

        return {"X": X}
