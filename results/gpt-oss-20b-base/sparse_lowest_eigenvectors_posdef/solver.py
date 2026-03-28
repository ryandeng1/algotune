import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        mat = problem["matrix"].asformat("csr")
        k = int(problem["k"])
        n = mat.shape[0]

        # If we need all or almost all eigenvalues, just use dense routine
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray(), subset_by_index=(0, k - 1))
            return [float(v) for v in vals]

        # Try sparse eigsh, fall back to dense if it fails
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SM",
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=min(n - 1, max(2 * k + 1, 20)),
            )[0]
        except Exception:
            vals = np.linalg.eigvalsh(mat.toarray(), subset_by_index=(0, k - 1))
        else:
            vals = np.sort(vals)

        return [float(v) for v in vals]