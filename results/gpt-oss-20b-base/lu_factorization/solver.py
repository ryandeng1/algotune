# solver.py

import numpy as np
from scipy.linalg import lu

class Solver:
    """
    Fast solver for LU factorisation problems.
    """

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Computes the LU decomposition A = P L U for a given square matrix `A`.

        Parameters
        ----------
        problem : dict[str, np.ndarray]
            Dictionary containing an entry 'matrix' mapping to a 2‑D NumPy array
            representing the matrix to factorise.

        Returns
        -------
        dict[str, dict[str, list[list[float]]]]
            Dictionary with a single key ``'LU'`` that contains the
            following nested dictionaries:
                * ``'P'`` – permutation matrix
                * ``'L'`` – lower triangular matrix
                * ``'U'`` – upper triangular matrix

            All matrices are converted to ordinary Python nested lists for
            compatibility with the test harness.
        """
        # Extract the input matrix
        A = problem["matrix"]

        # Direct numpy ‑ and scipy backed LU decomposition
        P, L, U = lu(A)

        # Convert to plain Python lists for the interface
        return {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}