import numpy as np
from scipy.linalg import svd


class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Solve the OPP instance by first computing M = B Aᵀ and
        computing its singular value decomposition: M = U S Vᵀ.
        The final solution is obtained via G = U Vᵀ.
        """
        A = problem.get("A")
        B = problem.get("B")
        if A is None or B is None or A.shape != B.shape:
            return {}

        # Ensure data is float64 for numeric stability
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)

        # Compute M = B @ A.T
        M = B @ A.T

        # Use SciPy's faster SVD routine
        U, _, Vt = svd(M, full_matrices=False)

        # Orthogonal solution G = U @ Vᵀ
        G = U @ Vt

        # Return as list of lists for the expected format
        return {"solution": G.tolist()}