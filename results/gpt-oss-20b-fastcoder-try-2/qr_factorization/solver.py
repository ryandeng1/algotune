import numpy as np
from scipy.linalg import qr

class Solver:
    """
    Solves A = Q R via the Lapack QR routine for maximum speed.
    """

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Compute the reduced QR factorisation of a matrix.

        Parameters
        ----------
        problem : dict[str, np.ndarray]
            A dictionary with a key ``matrix`` containing the matrix `A`.

        Returns
        -------
        dict
            A dictionary with a single key ``QR`` whose value is another dictionary
            containing the matrices ``Q`` and ``R`` as lists of lists.
        """
        A = problem["matrix"].copy()                # avoid modifying caller's data
        # ``overwrite_a`` speeds up by reusing the storage of A
        Q, R = qr(A, mode="economic", overwrite_a=True, check_finite=False)
        return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}