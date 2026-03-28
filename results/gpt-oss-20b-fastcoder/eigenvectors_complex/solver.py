import numpy as np
from numpy.typing import NDArray
from typing import List

class Solver:
    def solve(self, problem: NDArray) -> List[List[complex]]:
        """
        Solve the eigenvector problem for the given non‑symmetric matrix.
        Eigenvalues and eigenvectors are obtained via np.linalg.eig.
        Eigenpairs are sorted in descending order by the real part (then imaginary part)
        of the eigenvalues.  The returned eigenvectors are column‑normalized.
        """
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eig(problem)

        # Create ordering: sort by real, then imag (both descending)
        order = np.lexsort((-eigvals.imag, -eigvals.real))

        # Reorder eigenvectors according to the sorting
        sorted_vecs = eigvecs[:, order]

        # Normalize columns to unit length (avoid division by zero)
        norms = np.linalg.norm(sorted_vecs, axis=0)
        nonzero = norms > 1e-12
        sorted_vecs[:, nonzero] /= norms[nonzero]

        # Convert to list of lists of complex numbers
        return [col.tolist() for col in sorted_vecs.T]