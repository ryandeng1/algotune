from typing import Any, Dict, List
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, List[List[float]]]:
        """
        Solve the Orthogonal Procrustes Problem (OPP) via
        SVD of M = B Aᵀ and returning G = U Vᵀ.
        """
        # Extract and validate inputs
        A = problem.get("A")
        B = problem.get("B")
        if A is None or B is None:
            return {}

        # Ensure arrays and same shape
        A = np.asarray(A, dtype=np.float64, order="C")
        B = np.asarray(B, dtype=np.float64, order="C")
        if A.shape != B.shape:
            return {}

        # Compute M = B @ Aᵀ efficiently
        M = B.dot(A.T)

        # SVD with reduced matrices for speed
        U, _, Vt = np.linalg.svd(M, full_matrices=False)

        # Compute G = U @ Vt
        G = U @ Vt

        # Convert to nested Python list for return
        return {"solution": G.tolist()}