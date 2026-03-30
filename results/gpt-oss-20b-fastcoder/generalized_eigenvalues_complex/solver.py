from typing import Any, List
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

class Solver:
    """
    Solver for the generalized eigenvalue problem A x = λ B x.
    """

    def solve(self, problem: tuple[NDArray, NDArray]) -> List[complex]:
        """
        Solve the generalized eigenvalue problem for the given matrices A and B.

        The problem is defined as: A · x = λ B · x.
        For better numerical stability, B is scaled so that its Frobenius norm
        becomes one before solving the problem.  The resulting eigenvalues are
        sorted in descending order, first by real part then by imaginary part.

        Parameters
        ----------
        problem : tuple[NDArray, NDArray]
            A tuple of two real n×n matrices (A, B).

        Returns
        -------
        List[complex]
            List of eigenvalues sorted by real part descending, then
            imaginary part descending.
        """
        A, B = problem

        # Compute the Frobenius norm of B once
        scale_B = np.sqrt(np.linalg.norm(B, 'fro'))
        if scale_B == 0:  # avoid division by zero for a zero matrix
            return []

        # Scale both matrices (in a lightweight fashion)
        B_scaled = B / scale_B
        A_scaled = A / scale_B

        # Solve the generalized eigenvalue problem
        eigenvalues, _ = la.eig(A_scaled, B_scaled)

        # Sort by real part descending, then by imaginary part descending
        return sorted(eigenvalues, key=lambda x: (-x.real, -x.imag))