import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        """
        Compute the singular value decomposition of matrix A using NumPy's highly‑optimized
        LAPACK bindings. The `full_matrices=False` option and `check_finite=False` flag
        keep the routine lightweight, especially for large dense matrices.
        """
        A = problem["matrix"]
        # Ensure the data is contiguous and in the native NumPy format
        A = np.ascontiguousarray(A)
        # Compute U, S and Vh (V transposed)
        U, S, Vh = np.linalg.svd(A, full_matrices=False, check_finite=False)
        # Transpose Vh to obtain V
        V = Vh.T
        return {"U": U, "S": S, "V": V}