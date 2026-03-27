import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> list[complex]:
        A, B = problem

        # Scale matrices for better numerical stability
        scale = np.sqrt(np.linalg.norm(B, ord='fro'))
        if scale:
            A_scaled = A / scale
            B_scaled = B / scale
        else:
            A_scaled = A
            B_scaled = B

        # Solve the generalized eigenvalue problem
        eig_vals = la.eigvals(A_scaled, B_scaled)

        # Sort by real part (desc) then imaginary part (desc)
        idx = np.lexsort((-eig_vals.imag, -eig_vals.real))
        return eig_vals[idx].tolist()