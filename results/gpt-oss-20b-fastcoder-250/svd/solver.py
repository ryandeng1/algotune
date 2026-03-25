# solver.py
import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Compute the singular value decomposition of the given matrix.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
            - "n": number of rows
            - "m": number of columns
            - "matrix": the matrix as a list of lists or ndarray

        Returns
        -------
        dict
            Dictionary with keys:
            - "U": left singular vectors (n × k)
            - "S": singular values (k,)
            - "V": right singular vectors (m × k)
        """
        # Extract matrix
        A = np.asarray(problem["matrix"], dtype=np.float64)
        # Compute SVD with reduced shapes
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        V = Vh.T  # Convert Vh to V
        return {"U": U, "S": s, "V": V}
