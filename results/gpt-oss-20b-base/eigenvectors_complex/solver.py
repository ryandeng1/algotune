import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> list[list[complex]]:
        """Return normalized eigenvectors of a non‑symmetric matrix,
        sorted by eigenvalue real part descending, then imaginary part."""
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eig(problem)

        # Sorting indices: first by real part descending, then by imag part descending
        sort_idx = np.lexsort((
            -eigvals.imag,          # secondary key (imaginary)
            -eigvals.real,          # primary key (real)
        ))

        # Normalize eigenvectors column‑wise
        norms = np.linalg.norm(eigvecs, axis=0)
        norms[norms < 1e-12] = 1.0  # avoid division by zero
        eigvecs_normalized = eigvecs / norms

        # Order according to sort_idx
        sorted_evecs = eigvecs_normalized[:, sort_idx].T  # transpose to get list of vectors

        # Convert to list of lists of complex numbers
        return [vec.tolist() for vec in sorted_evecs]