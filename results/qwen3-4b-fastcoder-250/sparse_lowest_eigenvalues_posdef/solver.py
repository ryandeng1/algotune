import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        mat = problem["matrix"]
        k = int(problem["k"])
        n = mat.shape[0]
        
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v.real) for v in vals[:k]]
        else:
            try:
                vals = eigsh(
                    mat,
                    k=k,
                    which="SM",
                    return_eigenvectors=False,
                    maxiter=n * 200,
                    ncv=min(n - 1, max(2 * k + 1, 20)),
                )
                return [float(v.real) for v in vals[:k]]
            except Exception:
                vals = np.linalg.eigvalsh(mat.toarray())[:k]
                return [float(v.real) for v in vals]
