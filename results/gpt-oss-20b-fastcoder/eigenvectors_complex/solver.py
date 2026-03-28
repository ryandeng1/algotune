import numpy as np
from typing import List

class Solver:
    def solve(self, problem: np.ndarray) -> List[List[complex]]:
        """
        Solve the eigenvector problem for a non-symmetric matrix.
        Returns the eigenvectors sorted by descending real part
        (and then imaginary part) of the eigenvalues, each normalized to unit norm.
        """
        # Compute eigenvalues and right eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(problem)

        # Indices to sort by descending real part, then descending imaginary part
        sort_idx = np.lexsort((-eigenvalues.imag, -eigenvalues.real))

        # Reorder vectors
        vectors = eigenvectors[:, sort_idx]

        # Normalize columns
        norms = np.linalg.norm(vectors, axis=0, keepdims=True)
        # Avoid division by zero; norms that are zero stay as zero vector
        inv_norms = np.where(norms != 0, 1.0 / norms, 1.0)
        vectors = vectors * inv_norms

        # Convert to list of lists (each column becomes a list)
        return [vectors[:, i].tolist() for i in range(vectors.shape[1])]