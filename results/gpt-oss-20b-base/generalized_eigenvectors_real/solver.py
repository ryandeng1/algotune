#!/usr/bin/env python3
"""
Optimal solver for generalized symmetric eigenvalue problems.

The implementation focuses on minimizing Python overhead and making use of
NumPy's highly optimized linear algebra kernels. In particular,
the routine is fully vectorised, avoids explicit matrix inversions where
possible, and eliminates redundant control‑flow constructs.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import solve as solve_np
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular

class Solver:
    """Solver for the generalized eigenvalue problem  A·x = λ B·x with
    symmetric matrices A and positive‑definite symmetric matrix B.

    The algorithm follows the standard transformation to the
    ordinary eigenvalue problem:

        B = L·Lᵀ   (Cholesky factorisation)
        Ã = L⁻¹ A L⁻ᵀ

    The ordinary eigenproblem Ã·y = λ y is solved with `numpy.linalg.eigh`.
    Back‑transformation of eigenvectors together with the proper
    normalisation gives the requested results.
    """

    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> tuple[list[float], list[list[float]]]:
        A, B = problem

        # 1.  Cholesky factorisation of B
        L = cholesky(B, lower=True)                 # B = L Lᵀ

        # 2.  Solve the ordinary eigenvalue problem
        #    Compute Atilde = L⁻¹ A L⁻ᵀ without forming any inverses explicitly
        LinvA = solve_triangular(L, A, lower=True)   # L⁻¹ A
        Atilde = solve_triangular(L.T, LinvA, lower=False)  # L⁻¹ A L⁻ᵀ

        evals, evecs = np.linalg.eigh(Atilde)        # ascending order

        # 3.  Back‑transform eigenvectors
        #    y = evecs   (columns), x = Lᵀ y
        x = solve_triangular(L.T, evecs, lower=False)

        # 4.  Normalise eigenvectors:  vᵀ B v = ||L v||²
        #    Compute L v via forward substitution and then its norm.
        #    This is more accurate than computing vᵀ B v directly.
        norms = np.linalg.norm(solve_triangular(L, x, lower=True), axis=0)
        if np.any(norms == 0):
            # Avoid division by zero; zero vectors remain unchanged.
            norms = np.where(norms == 0, 1.0, norms)
        x /= norms

        # 5.  Sort in descending order of eigenvalues
        rev = np.arange(len(evals)-1, -1, -1)
        evals_desc = evals[rev]
        x_desc = x[:, rev]

        # 6.  Convert to plain Python objects
        eigenvalues = evals_desc.tolist()
        eigenvectors = [list(col) for col in x_desc.T]

        return eigenvalues, eigenvectors