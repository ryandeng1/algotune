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
        eigenvalues = np.linalg.eig(problem)[0]
        solution = sorted(eigenvalues, key=lambda x: (-x.real, -x.imag))
        return solution
