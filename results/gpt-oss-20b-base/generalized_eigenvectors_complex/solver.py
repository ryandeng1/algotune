import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> tuple[list[complex], list[list[complex]]]:
        """
        Solve the generalized eigenvalue problem A · x = λ B · x for real matrices A,B (n×n).
        Return eigenvalues sorted by descending (real, imag) part and matching unit‑norm eigenvectors.
        """
        A, B = problem

        # Scale only if B is not the identity to avoid unnecessary operations.
        scale_B = np.sqrt(np.linalg.norm(B, ord=2))
        if scale_B > 0.0:
            A_s, B_s = A / scale_B, B / scale_B
        else:
            A_s, B_s = A, B

        eigenvalues, eigenvectors = la.eig(A_s, B_s, left=False, right=True)

        # Normalise all eigenvectors in one vectorised operation
        norms = np.linalg.norm(eigenvectors, axis=0, keepdims=True)
        # Avoid division by zero – if a norm is zero, leave the vector unchanged
        safe_norms = np.where(norms == 0, 1, norms)
        eigenvectors = eigenvectors / safe_norms

        # Sort by descending real part, then descending imaginary part
        idx = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Convert to Python lists as required by the interface
        eigenvalues_list = list(eigenvalues)
        eigenvectors_list = [list(vec) for vec in eigenvectors.T]

        return eigenvalues_list, eigenvectors_list