import numpy as np
from typing import Any, Dict


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the SVD of the given matrix using NumPy's fast LAPACK routines."""
        A = problem["matrix"]
        # Fast SVD with minimal overhead
        U, s, Vh = np.linalg.svd(
            A,
            full_matrices=False,
            compute_uv=True,
            check_finite=False,  # Skip purity checks for speed
            lapack_driver="gesdd",  # Guaranteed best performance
        )
        V = Vh.T
        # Convert to native Python lists only once
        return {"U": U.tolist(), "S": s.tolist(), "V": V.tolist()}