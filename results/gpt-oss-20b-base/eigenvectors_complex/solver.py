import numpy as np
from numpy.typing import NDArray
from typing import List


class Solver:
    def solve(self, problem: NDArray) -> List[List[complex]]:
        """
        Solve the eigenvector problem for the given non‑symmetric matrix.

        Compute eigenvalues and eigenvectors, sort eigenpairs by descending
        real part (and then imaginary part) of the eigenvalues, normalize the
        eigenvectors to unit norm and return them as a list of lists of complex
        numbers.
        """
        A = problem
        eigvals, eigvecs = np.linalg.eig(A)

        # Obtain indices that order the eigenvalues by descending real part,
        # then imaginary part.
        order = np.lexsort((-eigvals.imag, -eigvals.real))
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Normalize each eigenvector
        norms = np.linalg.norm(eigvecs, axis=0, keepdims=True)
        # Avoid division by zero: set zero norm to 1 to keep vector unchanged
        norms[norms == 0] = 1.0
        eigvecs_norm = eigvecs / norms

        # Convert to list of lists
        return [vec.tolist() for vec in eigvecs_norm.T]