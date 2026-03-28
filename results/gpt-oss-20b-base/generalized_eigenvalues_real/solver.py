import numpy as np
from numpy.typing import NDArray
from typing import Any

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> list[float]:
        """
        Solve the generalized symmetric-definite eigenproblem A·x = λ·B·x.

        Uses a Cholesky transform without forming matrix inverses and
        relies on numpy's fast `eigvalsh` routine to compute eigenvalues of
        the transformed symmetric matrix.  The result is returned as a
        Python list sorted in descending order.
        """
        A, B = problem

        # Cholesky factorization B = L·Lᵀ
        L = np.linalg.cholesky(B)

        # Compute L^{-1}·A efficiently
        # solve(L, A) solves L·X = A for X, i.e. X = L^{-1}·A
        LA = np.linalg.solve(L, A)

        # Compute L^{-1}·A·L^{-T} = (L^{-1}·A)·L^{-T}
        # solve(L.T, LA.T) solves Lᵀ·Xᵀ = LAᵀ for Xᵀ, i.e. Xᵀ = L^{-T}·LAᵀ
        # so the transformed matrix is (solve(L.T, LA.T))ᵀ
        Atilde = np.linalg.solve(L.T, LA.T).T

        # Eigenvalues of a symmetric matrix
        eigenvalues = np.linalg.eigvalsh(Atilde)

        # Return in descending order as a plain Python list
        return list(eigenvalues[::-1])