import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray
from typing import Any, Tuple, List


class Solver:
    def solve(self, problem: Tuple[NDArray, NDArray]) -> Tuple[List[complex], List[List[complex]]]:
        """
        Solve the generalized eigenvalue problem A·x = λ B·x and return
        eigenvalues sorted by descending real part (then imaginary part) and
        unit‑norm eigenvectors.
        """
        A, B = problem
        n = A.shape[0]

        # Solve the generalized eigenproblem directly (scales automatically)
        eigvals, eigvecs = la.eig(A, B, type=1, overwrite_a=True, overwrite_b=True)

        # Normalize eigenvectors to unit norm
        norms = np.linalg.norm(eigvecs, axis=0, keepdims=True)
        # Avoid division by zero for zero‑norm columns
        zero_mask = norms.squeeze() <= 1e-15
        if zero_mask.any():
            norms[zero_mask] = 1.0
        eigvecs /= norms

        # Sort by descending real part, then descending imaginary part
        order = np.lexsort((-eigvals.imag, -eigvals.real))
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Convert to Python lists
        eigenvalues_list = eigvals.tolist()
        eigenvectors_list = [eigvecs[:, i].tolist() for i in range(n)]

        return eigenvalues_list, eigenvectors_list