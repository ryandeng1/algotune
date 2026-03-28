import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> list[complex]:
        """
        Solve the generalized eigenvalue problem for the given matrices A and B.

        The problem is defined as: A · x = λ B · x.
        Eigenvalues are computed directly from ``scipy.linalg.eigvals`` for better speed.
        The result list is sorted in descending order by real part, then imaginary part.

        :param problem: Tuple (A, B) with n × n real matrices.
        :return: Sorted list of eigenvalues (complex numbers).
        """
        A, B = problem
        # Compute only eigenvalues of the generalized problem.
        eigenvalues = la.eigvals(A, B)
        # Sort by descending real part, then descending imaginary part.
        return sorted(eigenvalues.tolist(), key=lambda x: (-x.real, -x.imag))