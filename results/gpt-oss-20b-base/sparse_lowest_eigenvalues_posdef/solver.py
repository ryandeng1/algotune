from typing import Any
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        """
        Return the k smallest eigenvalues of the symmetric matrix provided in
        problem["matrix"].  The matrix can be either a sparse matrix or a
        NumPy array that can be converted to CSR.
        """
        mat = problem["matrix"]
        if not isinstance(mat, csr_matrix):
            mat = mat.asformat("csr") if hasattr(mat, "asformat") else csr_matrix(mat)
        k = int(problem["k"])
        n = mat.shape[0]

        # if the requested number of eigenvalues is too large, fall back to dense
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # compute the k smallest eigenvalues with sparse eigsh
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SM",
                return_eigenvectors=False,
                maxiter=200 * n,
                ncv=min(n - 1, max(2 * k + 1, 20)),
            )[0]
        except Exception:
            # fallback to dense if something goes wrong
            vals = np.linalg.eigvalsh(mat.toarray())[:k]
        else:
            pass
        finally:
            pass

        return [float(v) for v in np.sort(np.real(vals))]