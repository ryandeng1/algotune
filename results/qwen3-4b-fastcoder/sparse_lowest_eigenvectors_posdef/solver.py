from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        mat = problem["matrix"].asformat("csr")
        k = int(problem["k"])
        n = mat.shape[0]

        if k >= n or n < 2 * k + 1:
            mat_dense = mat.toarray()
            vals = np.linalg.eigvalsh(mat_dense)
            return list(vals[:k])

        try:
            vals = eigsh(
                mat,
                k=k,
                which="SM",
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=min(n - 1, max(2 * k + 1, 20)),
            )
        except Exception:
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        return list(np.sort(vals))