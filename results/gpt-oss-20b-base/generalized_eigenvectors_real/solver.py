import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> tuple[list[float], list[list[float]]]:
        """
        Solve the generalized eigenvalue problem A·x = λ B·x.
        Uses a Cholesky-based transformation and efficient Numpy operations.
        """
        A, B = problem

        # Cholesky factorisation of B (B = L·Lᵀ)
        L = np.linalg.cholesky(B)

        # Transform the problem to a standard eigenvalue problem
        # Ã = L⁻¹ · A · L⁻ᵀ
        Linv = np.linalg.inv(L)                   #  smaller matrices so cost is acceptable
        Atilde = Linv @ A @ Linv.T

        # Solve the standard problem
        eigvals, eigvecs_tilde = np.linalg.eigh(Atilde)

        # Back‑transform the eigenvectors:
        #  x = L⁻ᵀ · y   (solve Lᵀ·x = y)
        eigvecs = np.linalg.solve(L.T, eigvecs_tilde)

        # Normalise eigenvectors so that vᵀ B v = 1
        # Compute all norms in one matrix product
        norms = np.sqrt(np.sum(eigvecs * (B @ eigvecs), axis=0))
        eigvecs = eigvecs / norms

        # Reverse order to obtain descending eigenvalues
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        # Convert to the required list format
        eigvals_list = eigvals.tolist()
        eigvecs_list = [eigvecs[:, i].tolist() for i in range(eigvecs.shape[1])]
        return eigvals_list, eigvecs_list