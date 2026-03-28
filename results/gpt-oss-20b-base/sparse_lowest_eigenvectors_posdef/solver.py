import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        mat = problem["matrix"].asformat("csr")
        k = int(problem["k"])
        n = mat.shape[0]

        # If almost all eigenvalues are needed or the matrix is tiny, fall back to dense
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Use sparse eigensolver for the smallest‑magnitude eigenvalues
        # eigsh returns values sorted in ascending order
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SM",
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=min(n - 1, max(2 * k + 1, 20)),
                tol=1e-10,
            )[0]
        except Exception:
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        return [float(v) for v in vals]