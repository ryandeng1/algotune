import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        """
        Return the k smallest eigenvalues of a symmetric matrix.
        """
        mat: sparse.spmatrix = problem['matrix'].asformat('csr')
        n = mat.shape[0]
        k = int(problem['k'])

        # For very small matrices assemble dense and use fast LAPACK routine
        if n <= 2000:   # threshold can be tuned
            vals = np.linalg.eigvalsh(mat.toarray())[:k]
            return [float(v) for v in vals]

        # For larger matrices use the sparse eigsh routine
        # If k >= n, fall back to dense
        if k >= n:
            vals = np.linalg.eigvalsh(mat.toarray())[:k]
            return [float(v) for v in vals]

        # Ensure k < n and matrix is not too small
        ncv = min(n - 1, max(2 * k + 1, 20))
        maxiter = n * 200

        try:
            vals = eigsh(
                mat,
                k=k,
                which="SM",
                return_eigenvectors=False,
                maxiter=maxiter,
                ncv=ncv,
            )
        except Exception:
            # Fallback to dense routine if sparse fails
            vals = np.linalg.eigvalsh(mat.toarray())[:k]
        else:
            vals = np.sort(vals)

        return [float(v) for v in vals]