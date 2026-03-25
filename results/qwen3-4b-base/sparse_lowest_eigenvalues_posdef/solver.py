import numpy as np
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        mat = problem["matrix"]
        k = int(problem["k"])
        n = mat.shape[0]
        
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in np.sort(vals[:k])]
        else:
            try:
                vals = eigsh(
                    mat,
                    k=k,
                    which="SM",
                    return_eigenvectors=False,
                    maxiter=500,
                    ncv=10
                )
                return [float(v) for v in np.sort(vals)]
            except Exception:
                vals = np.linalg.eigvalsh(mat.toarray())[:k]
                return [float(v) for v in np.sort(vals)]
