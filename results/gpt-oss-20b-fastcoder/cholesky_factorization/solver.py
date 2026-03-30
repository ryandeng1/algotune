#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Solver for Cholesky factorisation.

The implementation displaces conversion to a nested‑list until the very
last step in order to avoid unnecessary Python‑object creation during
the numerically expensive factorisation.  Apart from that the
function is just a thin wrapper around ``numpy.linalg.cholesky``,
which is highly optimised and already calls BLAS/LAPACK under the hood.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Any


class Solver:
    """
    Cholesky solver.

    Public API
    ----------
    solve(problem) -> dict
        Computes a Cholesky decomposition of the symmetric positive‑definite
        matrix stored in ``problem["matrix"]`` and returns the result in the
        structure required by the unit tests.

    Notes
    -----
    * ``numpy.linalg.cholesky`` already uses the best available
      implementation on the running system (MKL, OpenBLAS, Atlas, …).
    * Only at the very end is the NumPy array converted to a list of
      lists.  This postponement reduces the amount of Python‑level work
      and keeps the core computation fast.
    """

    def _matrix_from_problem(self, problem: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Extract the matrix from the problem dictionary.
        The matrix is expected to be a square NumPy array of type float64.
        """
        mat = problem["matrix"]
        if mat.dtype != np.float64:
            mat = mat.astype(np.float64, copy=False)
        return mat

    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Dict[str, List[List[float]]]]:
        """
        Return the Cholesky factorisation of the matrix stored in ``problem``.

        Parameters
        ----------
        problem : dict[str, np.ndarray]
            Dictionary containing a single key ``"matrix"`` that holds a
            square, symmetric and positive‑definite NumPy array.

        Returns
        -------
        dict
            ``{
                "Cholesky": {
                    "L": <list of lists of float>
                }
            }``

        Examples
        --------
        >>> import numpy as np
        >>> solver = Solver()
        >>> B = np.array([[4.0, 2.0], [2.0, 3.0]])
        >>> res = solver.solve({"matrix": B})
        >>> isinstance(res["Cholesky"]["L"], list)
        True
        """
        A = self._matrix_from_problem(problem)
        # Core computation: fast, low‑level BLAS/LAPACK call
        L = np.linalg.cholesky(A)
        # Final conversion to nested list structure
        return {"Cholesky": {"L": L.tolist()}}