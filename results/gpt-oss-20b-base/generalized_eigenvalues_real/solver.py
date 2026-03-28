from typing import Any
import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> list[float]:
        """
        Solve the generalized eigenvalue problem A·x = λ B·x.
        A is symmetric, B is symmetric positive definite.
        Returns eigenvalues sorted in descending order.
        """
        A, B = problem

        # Cholesky factorisation of B: B = L Lᵀ
        L = np.linalg.cholesky(B)          # L lower‑triangular

        # Solve L X = A  →  X = L⁻¹ A  (avoid explicit inversion)
        X = np.linalg.solve(L, A)

        # Transform to a standard eigenvalue problem
        # Atilde = (L⁻¹ A) (L⁻¹)ᵀ = X @ X.T
        Atilde = X @ X.T

        # Compute eigenvalues of the symmetric matrix
        eigs = np.linalg.eigh(Atilde, lower=True, eigvals_only=True)

        # Return descending sorted eigenvalues
        return eigs[::-1].tolist()