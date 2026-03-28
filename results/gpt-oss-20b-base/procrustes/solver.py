import numpy as np
from typing import Dict

class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Dict[str, list]]:
        """
        Solve the OPP instance by computing the orthogonal matrix G = U V^T from
        the SVD of M = B Aᵀ. Uses NumPy's efficient routines and avoids
        unnecessary copies.
        """
        A = problem.get("A")
        B = problem.get("B")
        if A is None or B is None:
            return {}
        # Ensure the inputs are NumPy arrays (view if already ndarray)
        A = np.asarray(A, dtype=np.double, order="C")
        B = np.asarray(B, dtype=np.double, order="C")
        if A.shape != B.shape:
            return {}

        # Compute M = B * Aᵀ
        M = B @ A.T

        # Singular value decomposition with economical matrices
        U, _, Vt = np.linalg.svd(M, full_matrices=False)

        # Orthogonal solution G = U Vᵀ
        G = U @ Vt

        # Convert to plain list-of-lists for the expected output format
        return {"solution": G.tolist()}