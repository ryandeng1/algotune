import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> list[list[complex]]:
        """
        Solve the eigenvector problem for the given non-symmetric matrix.
        Compute eigenvalues and eigenvectors using np.linalg.eig.
        Sort the eigenpairs in descending order by the real part (and then imaginary part)
        of the eigenvalues. Return the eigenvectors (each normalized to unit norm)
        as a list of lists of complex numbers.
        """
        # Compute eigenvalues and right eigenvectors
        eigvals, eigvecs = np.linalg.eig(problem)

        # Get sorting indices: descending by real, then imaginary part
        sort_idx = np.lexsort((-eigvals.imag, -eigvals.real))

        # Reorder eigenvectors accordingly
        eigvecs_sorted = eigvecs[:, sort_idx]

        # Normalize each eigenvector to unit norm
        norms = np.linalg.norm(eigvecs_sorted, axis=0)
        # Avoid division by zero (unlikely for non‑zero eigenvectors)
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        eigvecs_normalized = eigvecs_sorted / safe_norms

        # Convert to list of lists of complex numbers
        return eigvecs_normalized.T.tolist()