# solver.py
from __future__ import annotations
from typing import Any, List
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[float]:
        # Extract matrix and k
        mat: sparse.spmatrix = problem["matrix"].asformat("csr")
        k = int(problem["k"])
        n = mat.shape[0]

        # For very small or nearly full k, fall back to dense eig
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Sparse Lanczos for smallest magnitude eigenvalues
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
            # Rare fallback to dense computation
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        # Return sorted real values
        return [float(v) for v in np.sort(np.real(vals))]
