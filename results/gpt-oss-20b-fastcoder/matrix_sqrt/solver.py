"""
solver.py

Optimized solver for the principal matrix square root problem.
Instead of always calling `scipy.linalg.sqrtm`, we first test whether
the input matrix is Hermitian and positive‑semidefinite.  In that case
the eigen‑decomposition (`np.linalg.eigh`) is much faster and numerically
stable.  If the matrix fails this test, we fall back to `scipy.linalg.sqrtm`,
which handles the general case.

The implementation also removes the superfluous `try/except/finally` block
present in the original code to reduce branching overhead.  All conversion
from NumPy arrays to the required nested-list format is done once, at the
end, which keeps the function memory‑efficient and fast.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List

try:
    # scipy is optional; we import lazily to avoid import cost when not needed
    import scipy.linalg
except Exception:  # pragma: no cover
    scipy = None


class Solver:
    """Solver that computes the principal matrix square root.

    The returned dictionary has the form:
        {
            "sqrtm": {
                "X": [[complex, …], …]
            }
        }
    """

    @staticmethod
    def _is_hermitian(A: NDArray[np.complex128 | np.float64], eps: float = 1e-10) -> bool:
        """Return True if A is Hermitian (or real symmetric)."""
        return np.allclose(A, A.conj().T, atol=eps)

    @staticmethod
    def _is_positive_semidefinite(A: NDArray[np.complex128 | np.float64]) -> bool:
        """Return True if A is positive semidefinite.

        We test this on the eigenvalues after confirming Hermitian symmetry.
        """
        # Using `eigh` guarantees real eigenvalues for Hermitian matrices.
        vals = np.linalg.eigvalsh(A)
        return np.all(vals >= -1e-8)

    def solve(self, problem: Dict[str, NDArray[np.complex128 | np.float64]]) -> Dict[str, Dict[str, List[List[complex]]]]:
        """Compute the principal matrix square root."""
        A = problem["matrix"]

        # Fast path for Hermitian positive‑semidefinite matrices.
        if self._is_hermitian(A) and self._is_positive_semidefinite(A):
            # eigh returns eigenvalues in ascending order.
            vals, vecs = np.linalg.eigh(A)
            # Numerical safety: clip tiny negative values to zero.
            vals = np.where(vals > 0, np.sqrt(vals), 0.0)
            X = vecs @ np.diag(vals) @ vecs.conj().T
        else:
            # General case: fall back to scipy.linalg.sqrtm if available.
            if scipy is None:  # pragma: no cover
                raise RuntimeError(
                    "scipy.linalg.sqrtm is required for non‑Hermitian matrices."
                )
            X = scipy.linalg.sqrtm(A, disp=False)

        # Convert to nested lists; NumPy does the heavy lifting.
        X_list = X.tolist()
        return {"sqrtm": {"X": X_list}}