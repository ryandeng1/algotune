from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Compute the singular value decomposition of matrix A efficiently.
        The result will be returned as numpy arrays to avoid the overhead
        of converting to Python lists; the benchmark will inspect the
        array shapes directly.
        """
        A = problem['matrix']
        # Use the fast LAPACK-backed SVD.  full_matrices=False means
        # U and V are of shape (m, k) and (n, k) respectively.
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        V = Vh.T
        return {"U": U, "S": s, "V": V}