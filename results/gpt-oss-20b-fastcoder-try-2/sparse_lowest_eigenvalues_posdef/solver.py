import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict) -> list[float]:
        # Expect problem['matrix'] to be a csr_matrix or convertible via .asformat()
        mat = problem['matrix'].asformat('csr')
        k = int(problem['k'])
        n = mat.shape[0]

        # Fast path: dense or trivial
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Sparse route: use eigsh with sane defaults
        ncv = min(n - 1, max(2 * k + 1, 20))
        # eigsh may fail on singular matrices; fallback to dense if that happens
        try:
            vals = eigsh(mat, k=k, which='SM', return_eigenvectors=False,
                         maxiter=n * 200, ncv=ncv, tol=1e-8)[0]
        except Exception:
            vals = np.linalg.eigvalsh(mat.toarray())[:k]
        return [float(v) for v in np.sort(np.real(vals))]