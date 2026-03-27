import numpy as np
from scipy.sparse.linalg import eigsh
from typing import Any, List

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[float]:
        """Compute the k smallest‑magnitude eigenvalues of a symmetric matrix.

        Parameters
        ----------
        problem : dict[str, Any]
            Should contain:
                - ``matrix`` : a SciPy sparse matrix with a ``asformat`` method
                - ``k``     : int, number of eigenvalues to return

        Returns
        -------
        List[float]
            Sorted list of the k smallest‑magnitude eigenvalues (real part only).
        """
        # Pull inputs out of the problem dictionary
        mat = problem["matrix"].asformat("csr")          # keep CSR to preserve sparsity
        k   = int(problem["k"])
        n   = mat.shape[0]

        # Dense shortcut for trivial cases
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Use the efficient Lanczos method for sparse symmetric matrices.
        # ``which='SM'`` asks for smallest magnitude eigenvalues.
        # We request real eigenvalues only by setting `return_eigenvectors=False`.
        # ``ncv`` is the dimension of the Krylov subspace; we choose it slightly
        # larger than `k` but less than `n` to avoid memory blow‑up.
        ncv = max(2 * k + 1, 20)
        if ncv >= n:
            ncv = n - 1  # cannot be equal to n for eigsh

        vals = eigsh(
            mat,
            k=k,
            which="SM",
            return_eigenvectors=False,
            maxiter=n * 200,
            ncv=ncv,
        )

        # The values returned by eigsh are sorted already.  Cast to float
        # and return the real part (which should be real for a symmetric
        # matrix, but we keep the real() guard for completeness).
        return [float(v.real) for v in np.sort(vals.real)]