import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        # Convert to CSR format for efficient access
        mat: sparse.spmatrix = problem['matrix'].asformat('csr')
        k = int(problem['k'])
        n = mat.shape[0]

        # For very small matrices or when k is large, use dense eigen solver
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Use sparse eigen solver for the k smallest eigenvalues
        vals = eigsh(
            mat,
            k=k,
            which="SM",
            return_eigenvectors=False,
            maxiter=n * 200,
            ncv=min(n - 1, max(2 * k + 1, 20))
        )[0]

        # Ensure sorted real values
        return [float(v.real) for v in np.sort(np.real(vals))]