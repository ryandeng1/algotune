# solver.py
"""
Optimised solver for matrix exponential using SciPy's efficient implementation.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.linalg import expm

# --------------------------------------------------------------------------- #
# Solver class
# --------------------------------------------------------------------------- #
class Solver:
    """
    Compute the matrix exponential of a given square matrix.

    The implementation leverages SciPy's highly optimised `expm` routine
    which internally uses a scaling and squaring algorithm combined with Pade
    approximations. This is substantially faster than a purely Python
    implementation for matrices of practical size.

    """

    @staticmethod
    def _matrix_to_list(matrix: np.ndarray) -> List[List[float]]:
        """
        Convert a NumPy array to a nested list of Python floats.  This
        serialisation step is deliberately isolated in a separate method
        so that it can be optimized or replaced by a compiled routine if
        needed in the future.
        """
        return matrix.tolist()

    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, List[List[float]]]:
        """
        Compute the matrix exponential of the supplied matrix.

        Parameters
        ----------
        problem : Dict[str, np.ndarray]
            Dictionary with a single key ``'matrix'`` mapping to a
            square NumPy array.

        Returns
        -------
        Dict[str, List[List[float]]]
            Dictionary containing the key ``'exponential'`` with the
            matrix exponential represented as a list of lists of floats.
        """
        # Validate input
        A = problem["matrix"]
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Input matrix must be square")

        # Compute the exponential using SciPy's efficient routine
        expA = expm(A)

        # Convert to plain Python lists for the expected return format
        return {"exponential": self._matrix_to_list(expA)}