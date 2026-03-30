# solver.py
from __future__ import annotations
import numpy as np

class Solver:
    """Efficient solver for the Cholesky factorization problem.

    The implementation relies on NumPy's highly optimised LAPACK
    wrapper for the `cholesky` routine.  Because the benchmark
    only measures the *Python* runtime of the `solve` method, the
    conversion from NumPy array to plain Python lists is kept
    minimal to avoid any unnecessary overhead.
    """

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Compute the Cholesky factorisation A = L Lᵀ.

        Parameters
        ----------
        problem :
            Dictionary with a single entry `'matrix'` containing a
            positive‑definite, symmetric NumPy array.

        Returns
        -------
        dict
            A dictionary of the form:
            {
                'Cholesky': {
                    'L': [[float, ...], ...]
                }
            }
        """
        # Retrieve the matrix directly – no copies are made.
        A = problem["matrix"]

        # Fast, BLAS/LAPACK backed Cholesky decomposition.
        L = np.linalg.cholesky(A)

        # Convert to a plain list of lists only once.
        return {"Cholesky": {"L": L.tolist()}}