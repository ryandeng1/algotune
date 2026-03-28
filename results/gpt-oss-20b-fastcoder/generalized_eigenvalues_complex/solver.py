import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> list[complex]:
        """
        Solve the generalized eigenvalue problem A·x = λ B·x and return eigenvalues
        sorted descending by real part and then by imaginary part.
        """
        A, B = problem
        # Scale matrices to avoid overflow/underflow
        scale_B = np.linalg.norm(B, ord='fro') ** 0.5
        B_scaled = B / scale_B
        A_scaled = A / scale_B

        # Compute eigenvalues using LAPACK routine
        w, _ = la.eig(A_scaled, B_scaled)

        # Sort by real part descending, then by imaginary part descending
        w.sort()
        # numpy sorts ascending; reverse for descending
        w = w[::-1]
        # Ensure correct ordering for equal real parts
        # We can use stable Python sorting for small arrays
        sorted_vals = sorted(
            w,
            key=lambda z: (-z.real, -z.imag)
        )
        return list(sorted_vals)