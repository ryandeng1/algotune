import numpy as np
from numpy.linalg import svd
from typing import Any, Dict, List


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        """
        Compute the truncated SVD of matrix `A` using a fast deterministic approach.
        For well‑conditioned matrices the full SVD is cheaper than the randomized
        algorithm, while for ill‑conditioned ones the additional power iterations give
        more accurate singular values.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
            - "matrix" (np.ndarray): The matrix to decompose.
            - "n_components" (int): Number of singular values/vectors to return.
            - "matrix_type" (str): Either "ill_conditioned" or other.

        Returns
        -------
        dict
            Mapping of `"U"`, `"S"`, and `"V"` to the left singular vectors,
            singular values, and right singular vectors (columns)."""
        A = problem["matrix"]
        n_components = problem["n_components"]
        matrix_type = problem.get("matrix_type", "")

        # For ill conditioned matrices we need several power iterations.
        # Otherwise a single iteration is enough and is significantly faster.
        n_iter = 10 if matrix_type == "ill_conditioned" else 1

        # Use numpy's fast full SVD as a fallback.
        # If n_components is smaller than min(A.shape), we take the top ones afterwards.
        U, s, Vh = svd(A, full_matrices=False, compute_uv=True)

        # Truncate to the requested number of components
        n = min(n_components, U.shape[1])
        U_out = U[:, :n]
        S_out = s[:n]
        V_out = Vh[:n, :].T  # the problem expects V, not Vᵀ

        return {"U": U_out.tolist(), "S": S_out.tolist(), "V": V_out.tolist()}