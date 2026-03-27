from typing import Any, List
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


class Solver:
    def solve(self, problem: dict[str, Any]) -> List[float]:
        """Return the `k` smallest magnitude eigenvalues of the given symmetric
        matrix.  The matrix is supplied in a SciPy sparse format and may also
        be a dense `numpy.ndarray`.  The routine falls back to a dense
        computation only when the system is very small.
        """
        mat = problem["matrix"]

        # Ensure a CSR matrix for efficient row slicing.
        if not sparse.isspmatrix_csr(mat):
            mat = sparse.csr_matrix(mat)

        k = int(problem["k"])
        n = mat.shape[0]

        # Dense fallback for tiny systems or trivial eigenvalue counts.
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())[:k]
            return [float(v) for v in vals]

        # Sparse Lanczos (shift‑invert disabled) – returns unsorted eigenvalues.
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SM",               # smallest magnitude
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=min(n - 1, max(2 * k + 1, 20)),
            )
        except Exception:
            # Last‑resort dense fallback (rare).
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        # eigsh may give results in arbitrary order.
        vals = np.sort(vals)

        return [float(v) for v in vals]