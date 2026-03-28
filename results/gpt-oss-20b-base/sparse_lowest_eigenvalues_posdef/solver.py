import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        mat: sparse.spmatrix = problem['matrix'].asformat('csr')
        k = int(problem['k'])
        n = mat.shape[0]

        # If k is large relative to n, fall back to dense eigendecomposition
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Small memory configuration for lanczos
        ncv = min(n - 1, max(2 * k + 1, 20))
        try:
            vals = eigsh(mat, k=k, which='SM',
                         return_eigenvectors=False,
                         maxiter=n * 200,
                         ncv=ncv,
                         tol=1e-8)
        except Exception:
            # Fall back to dense if sparse solver fails
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        return [float(v) for v in np.sort(np.real(vals))]