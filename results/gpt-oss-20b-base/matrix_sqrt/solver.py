# solver.py
from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm

class Solver:
    """
    Optimised matrix square root solver using :func:`scipy.linalg.sqrtm`.
    """

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[complex]]]]:
        """
        Computes the principal matrix square root X of the input matrix A such that
        X @ X = A.

        The method simply delegates to `sqrtm`; errors are propagated as an empty
        matrix result to keep the public API unchanged.

        Parameters
        ----------
        problem
            Dictionary containing the key ``"matrix"`` mapped to the matrix A.

        Returns
        -------
        dict
            Dictionary with key ``"sqrtm"`` containing sub‑dictionary ``{"X": …}``
            where the value is a list of lists of complex numbers representing
            the square root matrix.
        """
        A = problem["matrix"]
        try:
            # sqrtm returns a tuple (X, Z). We only need X.
            X, _ = sqrtm(A, disp=False)
        except Exception:
            return {"sqrtm": {"X": []}}
        else:
            # Convert numpy array to nested Python lists (complex numbers)
            return {"sqrtm": {"X": X.tolist()}}