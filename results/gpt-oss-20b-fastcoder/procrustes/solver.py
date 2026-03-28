import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Compute the orthogonal matrix G = U Vᵀ that minimizes the Frobenius norm
        of A - G B for matrices A and B of equal shape.
        """
        A, B = problem.get('A'), problem.get('B')
        if A is None or B is None or A.shape != B.shape:
            return {}

        # Ensure we have NumPy arrays and avoid unnecessary copies
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)

        # Compute the cross‐product M = B Aᵀ
        M = B @ A.T

        # Fast SVD decomposition (economy size)
        U, _, Vt = np.linalg.svd(M, full_matrices=False)

        # The optimal orthogonal transformation
        G = U @ Vt

        return {'solution': G.tolist()}