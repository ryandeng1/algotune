from typing import Any, List
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[float]:
        """
        Return the k smallest eigenvalues of a real symmetric matrix.
        The matrix is expected in a sparse format (any subclass of `sparse.spmatrix`).
        """
        mat: sparse.spmatrix = problem['matrix'].asformat('csr')
        k: int = int(problem['k'])
        n = mat.shape[0]

        # If k is large enough to make the dense computation cheaper, or the matrix is very small,
        # fall back to a dense eigendecomposition.
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray(), subset_by_index=(0, k - 1))
            return [float(v) for v in vals]

        # Compute the k smallest magnitude eigenvalues using the sparse routine.
        try:
            vals = eigsh(
                mat,
                k=k,
                which='SM',
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=min(n - 1, max(2 * k + 1, 20)),
            )
        except Exception:
            # Fallback to dense if the sparse routine fails
            vals = np.linalg.eigvalsh(mat.toarray(), subset_by_index=(0, k - 1))

        # Ensure real values and sort (eigsh already returns sorted, but we keep safety)
        vals = np.sort(np.real(vals))
        return [float(v) for v in vals[:k]]