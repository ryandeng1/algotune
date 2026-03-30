# solver.py
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray
from typing import Tuple, List, Any

class Solver:
    """
    Optimised solver for the generalized eigenvalue problem
    A · x = λ B · x.
    """
    def solve(self, problem: Tuple[NDArray, NDArray]) -> Tuple[List[complex], List[List[complex]]]:
        A, B = problem

        # Scale B to improve numerical stability
        scale = np.sqrt(np.linalg.norm(B, ord='fro'))
        A_scaled = A / scale
        B_scaled = B / scale

        # Solve the generalized eigenvalue problem
        eigenvalues, eigenvectors = la.eig(A_scaled, B_scaled)

        # Normalise eigenvectors to unit L2 norm in a vectorised way
        norms = np.linalg.norm(eigenvectors, axis=0, keepdims=True)
        # Guard against division by zero
        safe_norms = np.where(norms == 0, 1, norms)
        eigenvectors = eigenvectors / safe_norms

        # Sort by real part descending, then imaginary part descending
        idx = np.argsort(
            -eigenvalues.real, kind='mergesort'
        ).argsort()  # stable sort by real part asc -> we will reverse
        # We need a secondary key for imaginary part. Use lexsort with negative imag
        # Assemble pairs for sorting
        pairs = list(zip(eigenvalues, np.split(eigenvectors, eigenvectors.shape[1], axis=1)))
        pairs.sort(key=lambda p: (-p[0].real, -p[0].imag))
        sorted_eigenvalues, sorted_eigenvectors = zip(*pairs)

        # Convert to required Python list format
        eigenvalues_list: List[complex] = list(sorted_eigenvalues)
        eigenvectors_list: List[List[complex]] = [list(vec.flatten()) for vec in sorted_eigenvectors]

        return eigenvalues_list, eigenvectors_list