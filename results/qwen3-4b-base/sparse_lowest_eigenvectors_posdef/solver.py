from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        mat: sparse.spmatrix = problem["matrix"].asformat("csr")
        k: int = int(problem["k"])
        n = mat.shape[0]

        # Dense path for tiny systems or k too close to n
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return vals[:k].tolist()

        # Sparse Lanczos without shift‑invert
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

        return vals.tolist()