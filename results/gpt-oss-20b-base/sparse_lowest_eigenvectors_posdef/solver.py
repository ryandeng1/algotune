import numpy as np
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict) -> list[float]:
        # Convert to CSR once
        mat = problem["matrix"].asformat("csr")
        n = mat.shape[0]
        k = int(problem["k"])

        # Degenerate case: need all eigenvalues
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Use sparse eigenvalue routine with safety fallback
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SM",  # smallest magnitude
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=min(n - 1, max(2 * k + 1, 20)),
            )
        except Exception:
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        return [float(v) for v in np.sort(np.real(vals))]