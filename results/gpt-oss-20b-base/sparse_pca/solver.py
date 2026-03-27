from __future__ import annotations
from typing import Any
import numpy as np


class Solver:
    """
    Fast, deterministic sparse PCA solver.

    The original implementation used cvxpy to solve a convex relaxation
    of the sparse PCA problem.  While correct it is slow (external
    solver launch, Python <-> COM communication, etc.).  Here we
    exploit the closed‑form solution of the eigen‑decomposition of the
    covariance matrix and perform element‑wise L1 soft‑thresholding
    before normalisation.  This yields a good approximate solution in a
    few milliseconds even for large matrices.
    """

    def _soft_threshold(self, x: np.ndarray, lam: float) -> np.ndarray:
        """Element‑wise soft thresholding."""
        return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Return sparse principal components with explained variance."""
        # ------------------------------------------------------------------
        # 1. Extract parameters and build covariance matrix
        # ------------------------------------------------------------------
        cov = np.asarray(problem["covariance"], dtype=np.float64)
        n_components = int(problem["n_components"])
        lam = float(problem["sparsity_param"])
        n = cov.shape[0]

        # ------------------------------------------------------------------
        # 2. Spectral decomposition – keep only positive eigenvalues
        # ------------------------------------------------------------------
        eigvals, eigvecs = np.linalg.eigh(cov)
        pos_mask = eigvals > 0
        eigvals = eigvals[pos_mask]
        eigvecs = eigvecs[:, pos_mask]

        # Sort descending
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Number of components we actually compute
        k = min(len(eigvals), n_components)
        # Scale eigenvectors by sqrt eigenvalues
        B = eigvecs[:, :k] * np.sqrt(eigvals[:k])

        # ------------------------------------------------------------------
        # 3. L1 soft‑thresholding (sparse approximation)
        # ------------------------------------------------------------------
        X = self._soft_threshold(B, lam)

        # ------------------------------------------------------------------
        # 4. Normalise columns to unit euclidean norm
        # ------------------------------------------------------------------
        norms = np.linalg.norm(X, axis=0, keepdims=True)
        # Avoid division by zero – if a column is zero after thresholding,
        # keep it as zero
        norms[norms == 0] = 1.0
        components = X / norms

        # ------------------------------------------------------------------
        # 5. Compute explained variance for each component
        # ------------------------------------------------------------------
        explained = []
        for i in range(k):
            vec = components[:, i]
            var = float(vec.T @ cov @ vec)
            explained.append(var)

        # Return the sparse components in list format
        return {
            "components": components.tolist(),
            "explained_variance": explained,
        }