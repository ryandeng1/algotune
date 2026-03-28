from typing import Any
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve, eigh

class Solver:

    def solve(self, problem: tuple[NDArray, NDArray]) -> list[float]:
        """
        Compute the eigenvalues of the generalized symmetric-definite eigenvalue problem
        A · x = λ B · x  with A symmetric and B symmetric positive definite.
        The algorithm uses a Cholesky factorisation of B, solves the standard
        eigenvalue problem Ã = B⁻¹ A with a single call to scipy.linalg.solve,
        and finally sorts the eigenvalues in descending order.
        """
        A, B = problem

        # Solve B·X = A  →  X = B⁻¹ A  (B is symmetric positive‑definite)
        X = solve(B, A, assume_a='pos')

        # Standard eigenvalue problem: Ã= B⁻¹ A
        eigenvalues = eigh(X, eigvals_only=True)

        # Return eigenvalues sorted in descending order
        return sorted(eigenvalues.tolist(), reverse=True)