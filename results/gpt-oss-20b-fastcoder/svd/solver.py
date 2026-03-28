import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        """
        Compute the singular value decomposition of matrix A with high performance.
        """
        A = np.asarray(problem['matrix'], dtype=np.float64, order='C')
        U, s, Vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)
        # numpy returns Vh; return V (transpose of Vh) and keep U, s as numpy arrays
        return {'U': U, 'S': s, 'V': Vh.T}