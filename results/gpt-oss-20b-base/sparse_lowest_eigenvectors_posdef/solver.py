from __future__ import annotations
from typing import Any, List

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[float]:
        """
        Compute the smallest `k` eigenvalues of a symmetric positive‑semidefinite
        sparse matrix in CSR format.  The matrix is assumed to be real and
        symmetric.  The eigenvalues are returned as a sorted list of floats
        in ascending order.

        Parameters
        ----------
        problem : dict[str, Any]
            Dictionary with keys:
                - "matrix": a scipy.sparse matrix (CSR or convertible to CSR)
                - "k": integer, number of smallest eigenvalues to return

        Returns
        -------
        List[float]
            Sorted list of the `k` smallest eigenvalues.
        """
        # Load matrix and parameters
        mat: sparse.spmatrix = problem["matrix"].asformat("csr")  # type: ignore
        k: int = int(problem["k"])
        n = mat.shape[0]

        # Small or almost full system: fallback to dense routine
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Use sparse Lanczos (no shift‑invert).  eigsh is very fast for this.
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SM",
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=min(n - 1, max(2 * k + 1, 20)),
            )
        except Exception:
            # Rare failure: fall back to dense computation
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        # Ensure sorted and converted to python float
        return [float(v) for v in np.sort(np.real(vals))]
