# solver.py
from __future__ import annotations
import numpy as np

class Solver:
    """
    Solver for the Orthogonal Procrustes Problem (OPP).
    The algorithm is:
    1. Compute M = B @ A.T
    2. Compute the SVD of M:  M = U @ S @ Vt
    3. Return G = U @ Vt  (an orthogonal matrix)
    """

    @staticmethod
    def _safe_numpy_array(x, dtype=np.float64):
        """
        Convert the input to a NumPy array without extra copies if the input
        is already a NumPy array of the desired dtype and contiguity.
        """
        if isinstance(x, np.ndarray) and x.dtype == dtype:
            return np.ascontiguousarray(x)
        return np.asarray(x, dtype=dtype, order="C")

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Solve the OPP instance.

        Parameters
        ----------
        problem : dict
            Expected keys: 'A', 'B', each mapping to a 2‑D array‑like object.
            Both matrices must have the same shape.

        Returns
        -------
        dict
            A dictionary with a single key `'G'` whose value is the orthogonal
            matrix that best aligns A to B.
        """
        # Retrieve the matrices
        A = problem.get("A")
        B = problem.get("B")
        if A is None or B is None:
            return {"G": np.empty((0, 0), float)}

        # Convert to contiguous NumPy arrays (fast path for already good arrays)
        A = self._safe_numpy_array(A)
        B = self._safe_numpy_array(B)

        if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape:
            # shapes mismatch – return empty solution
            return {"G": np.empty((0, 0), float)}

        # Compute the cross‑covariance matrix M
        M = B @ A.T  # (n, n)

        # SVD – use reduced mode for speed and memory efficiency
        U, _, Vt = np.linalg.svd(M, full_matrices=False)

        # Construct the orthogonal matrix G
        G = U @ Vt

        return {"G": G}