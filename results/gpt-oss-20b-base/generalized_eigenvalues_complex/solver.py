import numpy as np
import scipy.linalg as la
from typing import Any, List, Tuple

class Solver:
    def solve(self, problem: Tuple[np.ndarray, np.ndarray], **kwargs) -> List[complex]:
        """
        Solve the generalized eigenvalue problem for real square matrices A and B:
                   A · x = λ B · x.

        Parameters
        ----------
        problem : tuple of ndarray
            Tuple (A, B) with both matrices of shape (n, n).

        Returns
        -------
        List[complex]
            List of n eigenvalues sorted in descending order by real part and then
            imaginary part.
        """
        A, B = problem

        # Compute generalized eigenvalues
        eigenvalues, _ = la.eig(A, B, left=False, right=False)

        # Sort: descending by real part, then imaginary part
        eigenvalues.sort(key=lambda x: (-x.real, -x.imag))

        # Convert to Python list for consistency
        return list(eigenvalues)
