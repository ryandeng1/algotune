# solver.py
from __future__ import annotations
from typing import Any, List

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    """
    Solver for computing the k smallest eigenvalues of a square sparse
    positive‑semidefinite matrix.  The solution is a list of floats
    (the eigenvalues) sorted in ascending order.
    """

    def solve(self, problem: dict[str, Any], **_: Any) -> List[float]:
        """
        Parameters
        ----------
        problem : dict
            Must contain ``"matrix"`` (a scipy.sparse matrix in any format)
            and ``"k"`` (int, the number of smallest eigenvalues to return).

        Returns
        -------
        list[float]
            k smallest eigenvalues, sorted in ascending order.
        """
        # 1. Grab inputs
        mat: sparse.spmatrix = problem["matrix"]
        k: int = int(problem["k"])
        n: int = mat.shape[0]

        # 2. Small or dense case – use dense routine for speed on tiny matrices
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # 3. Sparse Lanczos for general case
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SM",          # smallest magnitude eigenvalues
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=min(n - 1, max(2 * k + 1, 20)),
            )
        except Exception:
            # Rare failure – fall back to dense computation
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        # 4. Return the sorted eigenvalues as floats
        return [float(v) for v in np.sort(np.real(vals))]
