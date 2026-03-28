import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from typing import Any, List

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[float]:
        """Return the smallest `k` eigenvalues of a symmetric matrix ``mat``."""
        # Convert to CSR once
        mat: sparse.spmatrix = problem["matrix"].asformat("csr")
        k = int(problem["k"])
        n = mat.shape[0]

        # If all eigenvalues are requested or the matrix is too small for sparse eigensolver
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return vals[:k].astype(float).tolist()

        # Sparse eigenvalue solver; fallback to dense if it fails
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SM",
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=min(n - 1, max(2 * k + 1, 20)),
            )
            return np.sort(vals).astype(float).tolist()
        except Exception:
            vals = np.linalg.eigvalsh(mat.toarray())
            return vals[:k].astype(float).tolist()