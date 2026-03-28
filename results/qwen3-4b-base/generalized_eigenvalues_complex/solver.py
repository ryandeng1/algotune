from typing import Any
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> list[complex]:
        A, B = problem

        # Scale matrices for better numerical stability.
        scale_B = np.sqrt(np.linalg.norm(B))
        B_scaled = B / scale_B
        A_scaled = A / scale_B

        # Solve scaled problem.
        eigenvalues, _ = la.eig(A_scaled, B_scaled)

        # Optimize sorting using numpy argsort
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        key = np.column_stack((-real_parts, -imag_parts))
        indices = np.argsort(key, axis=0)
        sorted_eigenvalues = eigenvalues[indices]
        return sorted_eigenvalues.tolist()