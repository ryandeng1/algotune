import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> list[list[complex]]:
        """
        Solve the eigenvector problem for the given non-symmetric matrix.
        Compute eigenvalues and eigenvectors using np.linalg.eig.
        Sort the eigenpairs in descending order by the real part (and then imaginary part) of the eigenvalues.
        Return the eigenvectors (each normalized to unit norm) as a list of lists of complex numbers.
        """
        # Compute eigenvalues and eigenvectors
        vals, vecs = np.linalg.eig(problem)

        # Order indices by descending real part, then imaginary part
        idx = np.lexsort((-vals.real, -vals.imag))

        # Reorder vectors
        vecs = vecs[:, idx]

        # Normalize each eigenvector to unit norm
        norms = np.linalg.norm(vecs, axis=0, keepdims=True)
        # Avoid division by zero; zero-norm vectors stay zero
        safe_norms = np.where(norms == 0, 1, norms)
        vecs = vecs / safe_norms

        # Convert to list of lists
        return [vec.tolist() for vec in vecs.T]