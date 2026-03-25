from __future__ import annotations
from typing import Any, List

import numpy as np
from scipy.sparse.linalg import eigsh


class Solver:
    """
    Solver for the Sparse Eigenvalues problem.
    Computes the k smallest eigenvalues (by magnitude) of a real symmetric
    positive semi‑definite sparse matrix.
    """

    def solve(self, problem: dict[str, Any]) -> List[float]:
        """
        Parameters
        ----------
        problem : dict
            A dictionary containing:
                - "matrix": a scipy.sparse CSR matrix (real, symmetric, PSD)
                - "k":     desired number of smallest eigenvalues

        Returns
        -------
        List[float]
            The k smallest eigenvalues (by magnitude), sorted in
            ascending order.
        """
        # Extract the problem data
        mat = problem["matrix"]          # already a CSR sparse matrix
        k = int(problem["k"])
        n = mat.shape[0]

        # Handle trivial cases with dense eigenvalue computation
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Otherwise use sparse Lanczos (eigsh) for the smallest magnitude eigenvalues
        try:
            # eigsh may raise if convergence fails; fallback below
            vals = eigsh(
                A=mat,
                k=k,
                which="SM",
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=min(n - 1, max(2 * k + 1, 20)),
            )[0]
        except Exception:
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        # Return sorted real values
        return [float(v) for v in np.sort(np.real(vals))]


# --- End of solver.py --------------------------------------------------
