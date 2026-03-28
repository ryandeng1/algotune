import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> list[complex]:
        """
        Solve the generalized eigenvalue problem A · x = λ B · x.

        The solution is a list of eigenvalues sorted first by real part,
        then by imaginary part, both in descending order.
        """
        A, B = problem
        # Scale matrices to improve numerical stability
        scale = np.linalg.norm(B) ** 0.5
        B_scaled = B / scale
        A_scaled = A / scale

        # Compute eigenvalues
        eigs, _ = la.eig(A_scaled, B_scaled)

        # Efficiently sort by real part descending, then imaginary part descending
        # `np.lexsort` sorts by the last key first, so we provide (-imag, -real)
        idx = np.lexsort((-eigs.imag, -eigs.real))
        return eigs[idx].tolist()