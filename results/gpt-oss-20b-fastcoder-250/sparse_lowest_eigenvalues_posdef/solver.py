from typing import Any, List
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

# Slight helper to ensure usage of CSR without unnecessary copies
def _to_csr(mat: sparse.spmatrix) -> sparse.csr_matrix:
    if mat.format != "csr":
        return mat.asformat("csr")
    return mat

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[float]:
        mat: sparse.csr_matrix = _to_csr(problem["matrix"])
        k = int(problem["k"])
        n = mat.shape[0]

        # Dense fallback for trivial cases
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Sparse Lanczos
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

        return [float(v) for v in np.sort(np.real(vals))]