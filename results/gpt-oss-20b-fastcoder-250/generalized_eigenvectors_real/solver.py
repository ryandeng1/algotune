import numpy as np
from typing import Any, List, Tuple

class Solver:
    """
    Solver for the generalized real eigenvalue problem A x = λ B x, where
    A is symmetric and B is symmetric positive definite.  Returns
    eigenvalues sorted in descending order and B‑orthonormal eigenvectors.
    """
    def solve(self, problem: Tuple[np.ndarray, np.ndarray], **kwargs) -> Tuple[List[float], List[List[float]]]:
        A, B = problem

        # ----- Solve the generalized eigenproblem
        # 1. Cholesky factorization of B: B = L Lᵀ, L lower triangular.
        L = np.linalg.cholesky(B)                     # B = L @ L.T

        # 2. Transform to standard symmetric eigenvalue problem.
        Linv = np.linalg.inv(L)                       # L⁻¹
        A_tilde = Linv @ A @ Linv.T                   # Ã = L⁻¹ A L⁻T

        # 3. Solve the standard hermitian problem.
        eigvals, eigvecs = np.linalg.eigh(A_tilde)    # values asc, columns are eigenvectors

        # 4. Back‑transform eigenvectors: v = L⁻T w
        eigvecs = Linv.T @ eigvecs                    # shape (n, n)

        # 5. Normalize each vector with respect to B: ||v||_B = 1
        Bv = B @ eigvecs
        norm_factor = np.sqrt(np.sum(eigvecs * Bv, axis=0))  # shape (n,)
        eigvecs = eigvecs / norm_factor

        # 6. Reverse to get descending eigenvalues.
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        # ----- Convert to pure Python lists for the interface
        eigenvalues_list: List[float] = eigvals.tolist()
        eigenvectors_list: List[List[float]] = [eigvecs[:, i].tolist() for i in range(eigvecs.shape[1])]

        return eigenvalues_list, eigenvectors_list
