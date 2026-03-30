"""
solver.py

This implementation performs sparse principal component analysis by first
computing the leading eigenvectors of the covariance matrix, then enforcing
a sparsity constraint by zeroing out the smallest-magnitude entries
until the desired sparsity level is achieved.  No external solver libraries
are required; the routine relies only on NumPy for speed.
"""

import numpy as np
from typing import Dict, Any


class Solver:
    """
    Solve a sparse PCA problem.

    The input dictionary must contain:
        - 'covariance': a square 2‑D list or array.
        - 'n_components': number of components to compute.
        - 'sparsity_param': target proportion of non‑zero entries
          (between 0 and 1).

    The method returns a dictionary with the components and their
    explained variances.
    """

    def _sparsify(self, vec: np.ndarray, sparsity: float) -> np.ndarray:
        """
        Retain the largest absolute entries such that the total
        non‑zero proportion is at most *sparsity*.
        """
        # Number of entries allowed to be non‑zero
        n = vec.size
        keep = max(1, int(np.floor(sparsity * n)))
        # Find indices of largest abs entries
        idx = np.argpartition(np.abs(vec), -keep)[-keep:]
        mask = np.zeros_like(vec, dtype=bool)
        mask[idx] = True
        vec_sparsified = np.where(mask, vec, 0.0)
        return vec_sparsified

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        # ---- Input handling -------------------------------------------------
        A = np.asarray(problem["covariance"], dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("covariance must be a square matrix")
        n_components = int(problem["n_components"])
        sparsity_param = float(problem["sparsity_param"])
        if not (0.0 <= sparsity_param <= 1.0):
            raise ValueError("sparsity_param must be in [0, 1]")

        n = A.shape[0]
        # ---- Eigen decomposition ---------------------------------------------
        eigvals, eigvecs = np.linalg.eigh(A)      # ascending order
        # keep only positive eigenvalues
        pos = eigvals > 0
        eigvals = eigvals[pos]
        eigvecs = eigvecs[:, pos]
        # sort by descending eigenvalue
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        k = min(len(eigvals), n_components)

        # ---- Initialise components -------------------------------------------
        components = np.zeros((n, n_components))
        explained_variance = []

        for i in range(k):
            # eigenvector scaled by sqrt(eigenvalue)
            vec = eigvecs[:, i] * np.sqrt(eigvals[i])
            # enforce sparsity
            vec = self._sparsify(vec, sparsity_param)
            # normalize to unit L2 norm if not zero
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            components[:, i] = vec
            # explained variance
            var = float(vec @ A @ vec)
            explained_variance.append(var)

        # if the user requested more components than positive eigenvalues,
        # pad with zeros
        if k < n_components:
            # remaining columns are already zero; explained variance stays empty
            pass

        # ---- Result ----------------------------------------------------------
        return {
            "components": components.tolist(),
            "explained_variance": explained_variance,
        }