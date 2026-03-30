# solver.py
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from typing import Any, List

__all__ = ["Solver"]


class Solver:
    """
    Optimised solver for the low‐lying eigenvalues of a sparse symmetric matrix.
    The implementation deliberately keeps the API identical to the reference
    code while avoiding unnecessary dense conversions and simplifying the
    control flow for maximum speed.
    """

    @staticmethod
    def _dense_eigvals(mat: np.ndarray, k: int) -> List[float]:
        """
        Compute the first *k* eigenvalues of a dense matrix.
        """
        # `np.linalg.eigvalsh` is highly optimised in LAPACK.
        vals = np.linalg.eigvalsh(mat)
        return vals[:k].astype(float).tolist()

    def solve(self, problem: dict[str, Any]) -> List[float]:
        """
        Return the smallest *k* eigenvalues of a symmetric matrix embedded
        in `problem['matrix']`, which must support `.asformat('csr')`.

        Parameters
        ----------
        problem : dict
            ``{'matrix': sparse_matrix, 'k': int}``

        Returns
        -------
        List[float]
            Sorted list of the first *k* real eigenvalues.
        """
        # Extract CSR matrix directly; this is the format used by eigsh.
        mat_csr: sparse.spmatrix = problem["matrix"].asformat("csr")
        k: int = int(problem["k"])
        n = mat_csr.shape[0]

        # Edge‑case 1: too many eigenvalues or too small a matrix.
        if k >= n or n < 2 * k + 1:
            return self._dense_eigvals(mat_csr.toarray(), k)

        # Parameters tuned for typical sparse problems:
        ncv = min(n - 1, max(2 * k + 1, 20))
        maxiter = n * 200

        # Core eigsh call.
        try:
            # Use the Lanczos method (default) with shift‑invert not requested.
            vals = eigsh(
                mat_csr,
                k=k,
                which="SM",          # smallest magnitude
                return_eigenvectors=False,
                maxiter=maxiter,
                ncv=ncv,
            )
        except Exception:
            # Fallback to dense if eigsh fails (e.g. non‑positive def. matrix).
            return self._dense_eigvals(mat_csr.toarray(), k)

        # Ensure the returned values are sorted and cast to float.
        return np.sort(np.real(vals)).astype(float).tolist()