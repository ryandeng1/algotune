import numpy as np
from scipy.linalg import eigvals
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> list[complex]:
        """
        Solve the generalized eigenvalue problem A·x = λ B·x for real matrices A and B.
        The eigenvalues are returned sorted by descending real part, then descending
        imaginary part.
        """
        A, B = problem

        # Scale both matrices with the Frobenius norm of B for stability.
        scale_B = np.linalg.norm(B, ord='fro')
        if scale_B == 0:
            # In case B is zero we fall back on the ordinary eigenvalue problem.
            eig = np.linalg.eigvals(A)
        else:
            factor = 1.0 / scale_B
            A_s = A * factor
            B_s = B * factor
            eig = eigvals(A_s, B_s, check_finite=True)

        # Sort by real part descending, then imaginary part descending.
        inds = np.lexsort((-eig.imag, -eig.real))
        return eig[inds].tolist()