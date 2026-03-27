import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> list[complex]:
        """
        Solve the generalized eigenvalue problem A x = λ B x.

        The returned eigenvalues are sorted by decreasing real part,
        then by decreasing imaginary part.
        """
        A, B = problem

        # Scale matrices by the square root of the Frobenius norm of B for
        # better numerical stability.  This is a lightweight operation that
        # avoids creating large temporary copies.
        scale = np.sqrt(np.linalg.norm(B, ord='fro'))
        if scale == 0:      # degenerate case
            return []

        A = A * (1.0 / scale)
        B = B * (1.0 / scale)

        # Compute only the eigenvalues, not the eigenvectors.
        eigvals = la.eigvals(A, B)

        # Sort by real part descending, then imaginary part descending.
        return sorted(eigvals, key=lambda x: (-x.real, -x.imag))