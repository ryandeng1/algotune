import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        mat = problem['matrix'].asformat('csr')
        k = int(problem['k'])
        n, _ = mat.shape

        # If we need all or almost all eigenvalues, fallback to dense routine
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Use sparse eigensolver for the k smallest magnitude eigenvalues
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

        return [float(v) for v in np.sort(np.real(vals))]