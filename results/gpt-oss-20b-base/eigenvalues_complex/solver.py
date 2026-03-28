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
        # Compute eigenvalues only – faster than eig which also returns eigenvectors
        eigvals = np.linalg.eigvals(problem)
        # Sort: descending real part first, then descending imaginary part
        # Use lexsort with negative values for descending order
        inds = np.lexsort((-eigvals.imag, -eigvals.real))
        sorted_vals = eigvals[inds]
        return sorted_vals.tolist()