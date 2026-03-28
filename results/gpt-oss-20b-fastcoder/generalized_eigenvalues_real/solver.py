import numpy as np
from typing import Any
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> list[float]:
        """
        Solve the generalized eigenvalue problem A · x = λ B · x
        for symmetric A and symmetric positive‑definite B.

        The algorithm transforms the problem to a standard one
        using a Cholesky decomposition of B and solves it with
        np.linalg.eigh.  All matrix inversions are avoided by
        solving triangular systems directly for better speed
        and numerical stability.
        """
        A, B = problem
        # Cholesky factor L such that B = L @ L.T
        L = np.linalg.cholesky(B)

        # Compute X = inv(L) @ A efficiently via solving L X = A
        X = np.linalg.solve(L, A)

        # Compute Atilde = inv(L) @ A @ inv(L).T
        # Solve L.T Z = X.T  ⇒  Z = inv(L).T @ X.T
        Z = np.linalg.solve(L.T, X.T)

        # Atilde is the transpose of Z
        Atilde = Z.T

        # Eigenvalues of the standard problem (ascending order)
        eigenvalues = np.linalg.eigh(Atilde)[0]

        # Return in descending order as a Python list
        return eigenvalues[::-1].tolist()