from typing import Any
import numpy as np
from scipy.linalg import solve_toeplitz

class Solver:
    """
    Minimal, highly‑optimized solver for Toeplitz linear systems.

    The `solve` method accepts the problem dictionary describing a
    Toeplitz system `T x = b` and returns the solution `x` as a
    Python list.  All heavy lifting is done by SciPy's
    `solve_toeplitz`, which keeps the computational complexity
    at O(n²) and avoids unnecessary Python‑level loops or
    temporary copies.
    """

    def __init__(self) -> None:
        # No initialization needed; keeping the constructor for
        # API compatibility.
        pass

    def solve(self, problem: dict[str, list[float]]) -> list[float]:
        """
        Solve a Toeplitz linear system.

        Parameters
        ----------
        problem : dict[str, list[float]]
            Must contain keys ``'c'``, ``'r'`` and ``'b'`` with
            sequences of floats.  ``c``/``r`` are the first column
            and first row of the Toeplitz matrix, and ``b`` is the
            right‑hand side vector.

        Returns
        -------
        list[float]
            The solution vector `x`.
        """
        # Convert inputs to NumPy arrays without unnecessary copies.
        c = np.asarray(problem['c'], dtype=np.float64, order='C')
        r = np.asarray(problem['r'], dtype=np.float64, order='C')
        b = np.asarray(problem['b'], dtype=np.float64, order='C')

        # Directly call the highly‑optimized SciPy routine.
        x = solve_toeplitz((c, r), b)

        # Return a plain Python list – the minimal extra step
        # required by the challenge specification.
        return x.tolist()