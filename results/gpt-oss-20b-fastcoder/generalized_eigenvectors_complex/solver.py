from typing import Any
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> tuple[list[complex], list[list[complex]]]:
        """
        Solve the generalized eigenvalue problem A · x = λ B · x
        and return sorted eigenvalues together with corresponding unit-norm eigenvectors.
        """
        A, B = problem

        # Scale both matrices to reduce numerical problems
        scale = np.sqrt(np.linalg.norm(B))
        A_scaled = A / scale
        B_scaled = B / scale

        # Compute eigenpairs (A_over_B)
        eigvals, eigvecs = la.eig(A_scaled, B_scaled)

        # Normalise eigenvectors to unit Euclidean norm
        norms = np.linalg.norm(eigvecs, axis=0)
        # Avoid divide-by-zero (unlikely for legitimate problems)
        mask = norms > 1e-15
        eigvecs[:, mask] = eigvecs[:, mask] / norms[mask]

        # Sort by real part, then imaginary part, descending
        order = np.argsort(
            np.lexsort((-eigvals.imag, -eigvals.real)),
            kind="quicksort",
        )
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Convert to python lists
        eigvals_list = [complex(v) for v in eigvals]
        eigvecs_list = [list(row) for row in eigvecs.T]

        return eigvals_list, eigvecs_list