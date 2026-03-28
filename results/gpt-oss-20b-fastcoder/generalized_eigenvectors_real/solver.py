import numpy as np
from typing import Any, Tuple, List
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: Tuple[NDArray, NDArray]) -> Tuple[List[float], List[List[float]]]:
        """
        Solve the generalized eigenvalue problem A x = λ B x
        for symmetric A and symmetric positive‑definite B.
        Returns eigenvalues (descending) and corresponding eigenvectors.
        """
        A, B = problem
        n = A.shape[0]

        # Cholesky of B: B = L Lᵀ
        L = np.linalg.cholesky(B)

        # Form Ã = L⁻¹ A (L⁻¹)ᵀ  without explicit inversion
        inv_L = np.linalg.solve(L, np.eye(n))            # L⁻¹
        A_tilde = inv_L @ A @ inv_L.T

        # Solve the standard symmetric eigenproblem
        eigvals, eigvecs = np.linalg.eigh(A_tilde)

        # Recover the eigenvectors of the generalized problem
        V = inv_L.T @ eigvecs

        # Reverse order: descending eigenvalues
        eigvals = eigvals[::-1]
        V = V[:, ::-1]

        return (eigvals.tolist(), [V[:, i].tolist() for i in range(n)])