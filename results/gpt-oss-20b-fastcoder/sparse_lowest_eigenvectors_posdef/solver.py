from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        mat: sparse.spmatrix = problem['matrix'].asformat('csr')
        k: int = int(problem['k'])
        n = mat.shape[0]

        # If k is too large, fall back to full diagonalisation
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Try sparse eigenvalue solver first
        try:
            vals = eigsh(
                mat,
                k=k,
                which='SM',
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=min(n - 1, max(2 * k + 1, 20))
            )
        except Exception:
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        # Return sorted (real) eigenvalues
        return [float(v) for v in np.sort(np.real(vals))]