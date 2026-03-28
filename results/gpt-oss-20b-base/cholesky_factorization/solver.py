from typing import Any
import numpy as np

class Solver:
    """
    Optimised Cholesky solver using NumPy's highly tuned linalg routines.
    """

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Compute the Cholesky factorization A = L @ L.T for the input matrix `A`.

        Parameters
        ----------
        problem : dict[str, np.ndarray]
            Dictionary containing the key 'matrix' mapping to a symmetric positive
            definite NumPy array of shape (n, n).

        Returns
        -------
        dict[str, dict[str, list[list[float]]]]
            Dictionary with key 'Cholesky' mapping to another dictionary that has
            the key 'L' mapping to a plain Python list of lists representation of the
            lower triangular factor.
        """
        # Extract the matrix; assume it is already a NumPy array
        A = problem["matrix"]

        # Directly use the fast CuPy/NumPy implementation of Cholesky
        L = np.linalg.cholesky(A)

        # Convert to a nested list of floats – this is the lightest conversion
        return {"Cholesky": {"L": L.tolist()}}