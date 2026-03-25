import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        mat = problem["matrix"].asformat("csr")
        k = int(problem["k"])
        n = mat.shape[0]
        
        if k >= n or n < 2 * k + 1:
            dense_mat = mat.toarray()
            vals = np.linalg.eigvalsh(dense_mat)[:k]
        else:
            vals = eigsh(
                mat,
                k=k,
                which="SM",
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=min(n - 1, max(2 * k + 1, 20)),
            )
        
        return [float(v.real) for v in vals]
