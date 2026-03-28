import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        # Convert to CSR for efficient operations
        mat = problem['matrix'].asformat('csr')
        k = int(problem['k'])
        n = mat.shape[0]

        # Handle trivial cases quickly
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return vals[:k].astype(float).tolist()

        # Parameters for eigsh
        ncv = max(20, min(n - 1, 2 * k + 1))
        maxiter = n * 200

        try:
            vals = eigsh(mat, k=k, which='SM',
                         return_eigenvectors=False, maxiter=maxiter, ncv=ncv)
        except Exception:
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        return np.sort(np.real(vals)).astype(float).tolist()