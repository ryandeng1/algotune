# solver.py
"""
Optimised Solver for a symmetric positive‑definite generalized eigenvalue problem.

Problem:          A · x = λ B · x
Assumptions:      A and B are Hermitian (real symmetric) and B is positive definite.
Solution:        Compute the eigenvalues λ in descending order.

The implementation eliminates explicit matrix inversion and uses the highly optimised
`scipy.linalg.eigh` routine that solves generalized Hermitian‑definite eigenproblems
efficiently. All intermediate arrays are kept memory‑efficient and no sorting
step touches the large arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh  # Fast BLAS/LAPACK backed routine


class Solver:
    """A benchmark‐friendly generalized eigensolver."""

    def solve(self, problem: tuple[NDArray[np.float64], NDArray[np.float64]]) -> list[float]:
        """
        Compute all eigenvalues of A · x = λ B · x.

        Parameters
        ----------
        problem : tuple
            Tuple (A, B) where:
            * A is a Hermitian matrix (np.ndarray of dtype float64).
            * B is a Hermitian positive‑definite matrix (dtype float64).

        Returns
        -------
        list[float]
            Eigenvalues sorted in descending order.
        """
        A, B = problem
        # Direct generalized eigendecomposition via LAPACK.
        # eigh(A, b=B) returns eigenvalues in ascending order.
        eigvals = eigh(A, b=B, eigvals_only=True)
        # Rearrange to descending order
        return list(eigvals[::-1])