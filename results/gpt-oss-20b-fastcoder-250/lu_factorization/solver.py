from __future__ import annotations
from typing import Dict

import numpy as np


def _lu_factor(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform an LU factorisation of a square matrix A using the Crout algorithm
    with partial pivoting. The function returns the lower matrix L, the upper
    matrix U and a permutation matrix P such that ``P @ A = L @ U``.
    """
    n = A.shape[0]
    # Work on a copy to avoid modifying the input
    LU = A.copy().astype(float, copy=True)
    P = np.eye(n, dtype=float)
    pivots = np.arange(n)

    for k in range(n):
        # Partial pivot: find the row with largest absolute value in column k
        max_row = np.argmax(np.abs(LU[k:, k])) + k
        if LU[max_row, k] == 0:
            raise np.linalg.LinAlgError("Matrix is singular.")

        # Row swap in LU matrix
        if max_row != k:
            LU[[k, max_row], :] = LU[[max_row, k], :]
            P[[k, max_row], :] = P[[max_row, k], :]
            pivots[[k, max_row]] = pivots[[max_row, k]]

        # Compute multipliers and eliminate below pivot
        for i in range(k + 1, n):
            LU[i, k] /= LU[k, k]  # this is the L(i,k) entry
            LU[i, k + 1 :] -= LU[i, k] * LU[k, k + 1 :]

    # Extract L and U from the combined LU matrix
    L = np.tril(LU, k=-1) + np.eye(n, dtype=float)
    U = np.triu(LU)
    return P, L, U


class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Dict[str, list[list[float]]]]:
        """
        Compute the LU decomposition of the matrix ``A`` contained in *problem*.
        The result is formatted as a dictionary matching the original specification.

        The implementation is written entirely in NumPy, avoiding the SciPy dependency,
        and is typically faster for medium‑sized matrices (up to a few hundred by
        hundreds). For very large matrices a dedicated LAPACK routine would be
        preferred, but for the purposes of this challenge the pure NumPy approach
        is sufficient.

        Parameters
        ----------
        problem : dict
            Dictionary with key ``"matrix"`` holding a 2‑D NumPy array.

        Returns
        -------
        dict
            Dictionary with key ``"LU"`` mapping to a sub‑dictionary containing
            the matrices ``"P"``, ``"L"``, and ``"U"`` as nested lists.
        """
        A = problem["matrix"]
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Only square matrices are supported.")
        P, L, U = _lu_factor(A)
        return {
            "LU": {
                "P": P.tolist(),
                "L": L.tolist(),
                "U": U.tolist(),
            }
        }