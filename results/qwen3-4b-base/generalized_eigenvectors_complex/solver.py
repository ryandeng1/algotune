from typing import Any
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> tuple[list[complex], list[list[complex]]]:
        A, B = problem

        # Scale matrices for better numerical stability
        scale_B = np.sqrt(np.linalg.norm(B))
        B_scaled = B / scale_B
        A_scaled = A / scale_B

        # Solve scaled problem
        eigenvalues, eigenvectors = la.eig(A_scaled, B_scaled)
        n = A.shape[0]

        # Vectorized normalization of eigenvectors
        norms = np.linalg.norm(eigenvectors, axis=0)
        mask = norms > 1e-15
        eigenvectors = np.where(mask[:, np.newaxis], eigenvectors / norms[:, np.newaxis], eigenvectors)

        # Sort eigenvalues and eigenvectors
        pairs = [(eigenvalues[i], eigenvectors[:, i]) for i in range(n)]
        pairs.sort(key=lambda pair: (-pair[0].real, -pair[0].imag))
        sorted_eigenvalues, sorted_eigenvectors = zip(*pairs)

        # Convert to Python lists
        eigenvalues_list = list(sorted_eigenvalues)
        eigenvectors_list = [list(vec) for vec in sorted_eigenvectors]

        return (eigenvalues_list, eigenvectors_list)