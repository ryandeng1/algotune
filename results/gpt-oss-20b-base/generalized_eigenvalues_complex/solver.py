import numpy as np
from typing import Any, List, Tuple
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: Tuple[NDArray[np.float64], NDArray[np.float64]]) -> List[complex]:
        """
        Solve the generalized eigenvalue problem A·x = λ B·x for real matrices A and B.
        Returns the eigenvalues sorted by descending real part, then by descending imaginary part.
        """
        A, B = problem
        # Scale B and A by the Frobenius norm of B for numerical stability.
        scale = np.linalg.norm(B, ord='fro')
        if scale == 0:
            scale = 1.0
        A_scaled = A / scale
        B_scaled = B / scale

        # Solve the generalized eigenvalue problem using NumPy's linear algebra routine.
        eigvals, _ = np.linalg.eig(A_scaled, B_scaled)

        # Sort by real part descending, then imaginary part descending.
        # Negative values are used for descending order.
        idx = np.lexsort((-eigvals.imag, -eigvals.real))
        return list(eigvals[idx])