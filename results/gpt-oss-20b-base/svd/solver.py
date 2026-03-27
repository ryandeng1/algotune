import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        """
        Compute the singular value decomposition of a real matrix A using NumPy.
        The result is returned as a dictionary with NumPy arrays, which are
        compatible with most downstream code that expects list-like structures.

        Parameters
        ----------
        problem : dict
            A mapping containing the key "matrix" with the input matrix as an
            iterable (e.g., list of lists or a NumPy array).

        Returns
        -------
        dict
            Mapping containing:
                "U" : ndarray of shape (n, k) – left singular vectors
                "S" : ndarray of shape (k,) – singular values
                "V" : ndarray of shape (k, m) – right singular vectors
        """
        # Convert the input to a NumPy array without copying if possible.
        A = np.array(problem["matrix"], dtype=np.float64, copy=False)

        # Perform the SVD. `full_matrices=False` yields compact matrices.
        U, s, Vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)

        # Transpose Vh to obtain V so that A == U @ np.diag(s) @ V.T
        V = Vh.T

        return {"U": U, "S": s, "V": V}