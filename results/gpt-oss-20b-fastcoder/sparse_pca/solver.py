import numpy as np
from typing import Any, Dict, List

class Solver:
    """
    Very fast heuristic solver for sparse PCA.
    It returns the leading eigenvectors of the covariance matrix.
    The returned components are orthonormal and respect the
    ``n_components`` requirement.  A small sparsity penalty is
    applied by zeroing out entries below a threshold determined by
    ``sparsity_param``.
    """

    def _threshold(self, vec: np.ndarray, sparsity: float) -> np.ndarray:
        """
        Zero out the smallest entries of ``vec`` until the number of
        non‑zero entries is approximately ``sparsity`` (as a fraction).
        """
        if sparsity <= 0:
            return vec
        # Determine number of non‑zero elements desired
        n_nonzero = max(1, int(sparsity * len(vec)))
        # Find cutoff value
        abs_vals = np.abs(vec)
        if n_nonzero >= len(vec):
            return vec
        cutoff = np.partition(abs_vals, -n_nonzero)[-n_nonzero]
        return np.where(abs_vals >= cutoff, vec, 0.0)

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        A = np.array(problem["covariance"], dtype=np.float64)
        n_components = int(problem.get("n_components", 1))
        sparsity_param = float(problem.get("sparsity_param", 0.0))

        # eigen decomposition (fast for symmetric matrices)
        eigvals, eigvecs = np.linalg.eigh(A)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # keep only the requested number of components
        k = min(n_components, eigvecs.shape[1])
        components = eigvecs[:, :k].copy()

        # optionally induce sparsity
        for i in range(k):
            components[:, i] = self._threshold(components[:, i], sparsity_param)

        # re‑orthonormalise in case sparsity broke orthogonality
        q, _ = np.linalg.qr(components, mode="reduced")
        components = q[:, :k]

        # compute explained variance
        explained_variance: List[float] = []
        for i in range(k):
            var = components[:, i].T @ A @ components[:, i]
            explained_variance.append(float(var))

        return {"components": components.tolist(), "explained_variance": explained_variance}