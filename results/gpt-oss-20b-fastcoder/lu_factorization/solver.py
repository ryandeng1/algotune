# Optimised `solve` implementation
from typing import Any
import numpy as np
from scipy.linalg import lu_factor, lu_solve

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Solve an LU factorisation problem using scipy's `lu_factor` for speed.
        Calculates the decomposition A = P @ L @ U where:
          - P: permutation matrix (expressed as a 2‑D list)
          - L: lower triangular matrix with unit diagonal (2‑D list)
          - U: upper triangular matrix (2‑D list)

        Parameters
        ----------
        problem : dict
            Must contain a key 'matrix' with a 2‑D np.ndarray.

        Returns
        -------
        dict
            Nested dictionary structure containing the decomposition.
        """
        A = problem["matrix"]

        # Use LU factorisation with partial pivoting
        lu, piv = lu_factor(A)

        # Construct permutation matrix P from pivot indices
        n = A.shape[0]
        P = np.eye(n, dtype=lu.dtype)
        for i, pi in enumerate(piv):
            if pi != i:
                P[[i, pi]] = P[[pi, i]]

        # Extract L (unit diagonal) and U (upper triangular)
        L = np.tril(lu, k=-1) + np.eye(n, dtype=lu.dtype)
        U = np.triu(lu)

        return {
            "LU": {
                "P": P.tolist(),
                "L": L.tolist(),
                "U": U.tolist(),
            }
        }