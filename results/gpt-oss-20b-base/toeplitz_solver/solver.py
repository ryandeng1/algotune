import numpy as np
from scipy.linalg import solve_toeplitz

class Solver:
    def solve(self, problem: dict[str, list[float]]) -> list[float]:
        """
        Solve the linear Toeplitz system Tx = b.

        The input dictionary contains three keys:
          * 'c' – first column of the Toeplitz matrix T
          * 'r' – first row of T
          * 'b' – right‑hand side vector

        The implementation relies on `scipy.linalg.solve_toeplitz`, which
        performs the factorisation and solve in O(n²) time with a small
        constant factor, making it significantly faster than a generic
        linear solver for large systems.

        Parameters
        ----------
        problem : dict[str, list[float]]
            Problem specification.

        Returns
        -------
        list[float]
            Solution vector x.
        """
        # The fastest path: convert to 1‑D numpy arrays once,
        # then hand them to the underlying LAPACK routine via SciPy.
        c = np.asarray(problem['c'], dtype=np.float64, order='C', copy=False)
        r = np.asarray(problem['r'], dtype=np.float64, order='C', copy=False)
        b = np.asarray(problem['b'], dtype=np.float64, order='C', copy=False)

        # SciPy's `solve_toeplitz` accepts the Toeplitz data as a tuple.
        x = solve_toeplitz((c, r), b, overwrite_b=True, check_finite=False)

        # Convert back to Python list for the public API.
        return x.tolist()