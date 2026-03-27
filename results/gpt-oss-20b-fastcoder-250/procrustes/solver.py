from typing import Any
import numpy as np


class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Solve the OPP instance by computing M = B Aᵀ, its singular value decomposition
        M = U Σ Vᵀ, and returning the orthogonal matrix G = U Vᵀ.

        Parameters
        ----------
        problem : dict[str, np.ndarray]
            A dictionary containing the 'A' and 'B' matrices.

        Returns
        -------
        dict
            Dictionary containing a single entry 'solution' with the orthogonal matrix
            G represented as a nested list of floats.
        """

        # Retrieve input matrices
        A = problem.get("A")
        B = problem.get("B")

        # Quick sanity checks
        if A is None or B is None:
            return {}
        if A.shape != B.shape:
            return {}

        # Ensure NumPy arrays (avoid copying if already an array)
        if not isinstance(A, np.ndarray):
            A = np.array(A, dtype=float, copy=False)
        if not isinstance(B, np.ndarray):
            B = np.array(B, dtype=float, copy=False)

        # Compute M = B @ Aᵀ
        M = B @ A.T

        # Perform an economy SVD to avoid unnecessary padding
        U, _, Vt = np.linalg.svd(M, full_matrices=False)

        # Compute the orthogonal solution G = U @ Vᵀ
        G = U @ Vt

        # Convert to nested list for the required output format
        return {"solution": G.tolist()}