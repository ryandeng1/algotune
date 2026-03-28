import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> list[list[complex]]:
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(problem)

        # Normalize all eigenvectors column‑wise (avoid loops)
        norms = np.linalg.norm(eigenvectors, axis=0)
        # Prevent division by zero
        norms_safe = np.where(norms > 1e-12, norms, 1)
        eigenvectors = eigenvectors / norms_safe

        # Sort indices by real part (desc), then imaginary part (desc)
        sorted_idx = np.lexsort((
            -eigenvalues.imag,
            -eigenvalues.real
        ))[::-1]

        # Reorder and convert to list of lists
        sorted_vectors = eigenvectors[:, sorted_idx]
        return [vec.tolist() for vec in sorted_vectors.T]