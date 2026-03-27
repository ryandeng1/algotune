from typing import Dict
import numpy as np


class Solver:
    """
    Solver that computes an orthogonal matrix G that best aligns two
    given matrices A and B in the Frobenius norm sense.
    """

    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Dict[str, list[list[float]]]]:
        A = problem.get("A")
        B = problem.get("B")
        if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
            return {}
        if A.shape != B.shape:
            return {}
        # Compute M = B Aᵀ
        M = B @ A.T
        # SVD of M
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        # Compute G = U Vᵀ
        G = U @ Vt
        return {"solution": G.tolist()}