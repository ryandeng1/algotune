import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> list[complex]:
        A, B = problem

        # Scale matrices for numerical stability
        norm_B = np.linalg.norm(B, ord='fro')
        scale = np.sqrt(norm_B) if norm_B != 0 else 1.0
        A_scaled = A / scale
        B_scaled = B / scale

        # Compute eigenvalues for the generalized problem
        eigvals, _ = la.eig(A_scaled, B_scaled)

        # Sort: first by real part (descending), then by imaginary part (descending)
        # np.lexsort sorts by the last key first
        idx = np.lexsort((-eigvals.imag, -eigvals.real))
        sorted_vals = eigvals[idx]

        return sorted_vals.tolist()