# solver.py
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    """
    Fast solver for the smallest k eigenvalues of a real symmetric matrix
    that is provided as a scipy sparse matrix.
    """
    _DEFAULT_MAX_ITER_MULTIPLIER = 200
    _MIN_NCV = 20

    def solve(self, problem: dict[str, Any]) -> list[float]:
        """
        Return the smallest *k* eigenvalues of the matrix associated with
        ``problem``.

        Parameters
        ----------
        problem
            Dictionary containing at least two keys:
                ``matrix``: a scipy sparse matrix (any format)
                ``k``      : integer number of eigenvalues to retrieve

        Returns
        -------
        list[float]
            Sorted list of the k smallest eigenvalues (real part only).
        """
        mat: sparse.spmatrix = problem["matrix"].asformat("csr")
        k = int(problem["k"])
        n = mat.shape[0]

        # Special cases where the dense eigvals are cheaper or necessary
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Prefer sparse eigsh.  For very small matrices it is better to
        # fall back to the dense algorithm.
        max_iter = n * self._DEFAULT_MAX_ITER_MULTIPLIER
        ncv = min(n - 1, max(2 * k + 1, self._MIN_NCV))

        vals = eigsh(
            mat,
            k=k,
            which="SM",
            return_eigenvectors=False,
            maxiter=max_iter,
            ncv=ncv,
            # nordering is not needed; eigsh returns sorted eigenvalues
        )
        # eigsh guarantees real eigenvalues for symmetric input.
        return [float(v) for v in vals]