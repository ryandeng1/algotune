# solver.py
from typing import List, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


class Solver:
    """Efficient solver for the generalized eigenvalue problem A·x = λ B·x."""

    __slots__ = ()

    def solve(self, problem: Tuple[NDArray, NDArray]) -> List[complex]:
        """
        Solve the generalized eigenvalue problem for matrices A and B.

        The solution is a list of eigenvalues sorted in descending order,
        first by real part and then by imaginary part.

        Parameters
        ----------
        problem : tuple[NDArray, NDArray]
            Tuple containing the square matrices A and B.

        Returns
        -------
        list[complex]
            Eigenvalues sorted by descending real part then imaginary part.
        """
        A, B = problem

        # Scale the matrices to improve numerical stability.
        # Frobenius norm is fast and works well for this purpose.
        scale = np.sqrt(np.linalg.norm(B, ord="fro"))
        # Inplace scaling keeps memory usage low.
        A = A / scale
        B = B / scale

        # Compute only eigenvalues – faster than full diagonalisation.
        eigenvalues = la.eigvals(A, B)

        # Sort by real part descending, then imag part descending.
        # Using numpy argsort is faster than Python's sorted.
        # Create a structured array for ordering.
        order = -eigenvalues.real - 1j * eigenvalues.imag
        idx = np.argsort(order, kind="quicksort")
        sorted_eigs = eigenvalues[idx.tolist()]

        # Convert to a plain Python list for the required return type.
        return sorted_eigs.tolist()