from typing import List
import numpy as np
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: NDArray) -> List[List[complex]]:
        """
        Solve the eigenvector problem for the given non‑symmetric matrix.
        Returns a list of unit‑norm eigenvectors sorted in descending order by
        (real part, then imaginary part) of the corresponding eigenvalues.
        """
        # Compute eigenvalues and right eigenvectors
        eigvals, eigvecs = np.linalg.eig(problem)

        # Sort indices by descending real part, then descending imaginary part
        order = np.lexsort(
            (
                -eigvals.imag,
                -eigvals.real,
            )
        )
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Normalize each eigenvector
        norms = np.linalg.norm(eigvecs, axis=0, keepdims=True)
        # Avoid division by zero (norm < 1e-12); keep zero vector unchanged
        safe_norms = np.where(norms < 1e-12, 1.0, norms)
        eigvecs = eigvecs / safe_norms

        # Convert to list of lists of complex numbers
        return [vec.tolist() for vec in eigvecs.T]