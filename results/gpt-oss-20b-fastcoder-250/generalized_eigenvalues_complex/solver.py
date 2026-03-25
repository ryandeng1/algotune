# solver.py

import numpy as np
import scipy.linalg as la

class Solver:
    def solve(self, problem: tuple[np.ndarray, np.ndarray], **kwargs) -> list:
        """
        Solve the generalized eigenvalue problem A·x = λ B·x.

        Parameters
        ----------
        problem : tuple[np.ndarray, np.ndarray]
            A tuple (A, B) where A and B are n×n real matrices.

        Returns
        -------
        list[complex]
            List of n eigenvalues sorted in descending order by real part
            and then by imaginary part.
        """
        A, B = problem

        # Scale B for numerical stability (as in the reference, but optional).
        norm_B = np.linalg.norm(B, ord='fro')
        if norm_B > 0:
            scale = np.sqrt(norm_B)
            A_scaled = A / scale
            B_scaled = B / scale
        else:
            A_scaled = A
            B_scaled = B

        # Solve the generalized eigenvalue problem.
        eigvals, _ = la.eig(A_scaled, B_scaled)

        # Sort by descending real part, then descending imaginary part.
        sorted_eigvals = sorted(eigvals, key=lambda x: (-x.real, -x.imag))
        return list(sorted_eigvals)
