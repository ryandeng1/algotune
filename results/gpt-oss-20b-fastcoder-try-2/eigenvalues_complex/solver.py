from typing import Any
import numpy as np
from numpy.typing import NDArray

class Solver:

    def solve(self, problem: NDArray) -> list[complex]:
        """
        Solve the eigenvalue problem for the given square matrix.
        The solution returned is a list of eigenvalues sorted in descending order.
        The sorting order is defined as follows: first by the real part (descending),
        then by the imaginary part (descending).
        :param problem: A numpy array representing the real square matrix.
        :return: List of eigenvalues (complex numbers) sorted in descending order.
        """
        # Compute eigenvalues only (no eigenvectors)
        eigs = np.linalg.eigvals(problem)

        # Sort by real part descending, then imaginary part descending
        # lexsort expects keys in ascending order, so we negate for descending
        keys = (-eigs.real, -eigs.imag)
        order = np.lexsort(keys)

        # Apply ordering and convert to a plain Python list
        return list(eigs[order])