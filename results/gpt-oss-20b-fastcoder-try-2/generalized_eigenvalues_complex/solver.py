from typing import Any, Tuple, List
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: Tuple[NDArray, NDArray]) -> List[complex]:
        """
        Solve the generalized eigenvalue problem A x = λ B x.
        
        1. Scale matrices for numerical stability.
        2. Compute eigenvalues directly using SciPy's eigvals routine.
        3. Sort by descending real part, then imaginary part.
        """
        A, B = problem

        # Scale both matrices by sqrt(frobenius norm of B)
        scale = np.sqrt(np.linalg.norm(B, ord='fro'))
        A_scaled = A / scale
        B_scaled = B / scale

        # SciPy's eigvals returns only eigenvalues
        vals = la.eigvals(A_scaled, B_scaled)

        # Sort: first by real part descending, then by imag part descending
        vals.sort(key=lambda x: (-x.real, -x.imag))

        return vals.tolist()