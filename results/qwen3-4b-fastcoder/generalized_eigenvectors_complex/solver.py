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

        # Vectorized normalization with mask
        norms = np.linalg.norm(eigenvectors, axis=0)
        eigenvectors = np.where(norms > 1e-15, eigenvectors / norms[:, np.newaxis], eigenvectors)

        # Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort( (-eigenvalues.real, -eigenvalues.imag) )
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Convert to Python lists
        eigenvalues_list = sorted_eigenvalues.tolist()
        eigenvectors_list = [vec.tolist() for vec in sorted_eigenvectors]

        return (eigenvalues_list, eigenvectors_list)