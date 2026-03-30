# solver.py
from typing import Any, Dict, List
import numpy as np
import cvxpy as cp


class Solver:
    """
    Sparse PCA solver.
    The implementation is deliberately lightweight:
    * All heavy linear‑algebra is done with NumPy.
    * The optimisation problem is set up in CVXPY as a single convex program.
    * Result extraction is fully vectorised to avoid repeated Python loops.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        """
        Solve a sparse PCA problem.

        Parameters
        ----------
        problem : dict
            Dictionary containing the keys
                - 'covariance'      : 2‑D float list or ndarray
                - 'n_components'    : int (requested number of components)
                - 'sparsity_param'  : float (regularisation weight)

        Returns
        -------
        dict
            Dictionary with keys
                - 'components'       : list of lists, shape (n_features, n_components)
                - 'explained_variance': list of variance explained by each component
        """
        # ---------------------------------------------------------------------
        # 1) Basic data extraction and preparation
        # ---------------------------------------------------------------------
        A = np.asarray(problem["covariance"], dtype=np.float64)  # covariance matrix
        n_components = int(problem["n_components"])
        sparsity = float(problem["sparsity_param"])

        n = A.shape[0]

        # ---------------------------------------------------------------------
        # 2) Compute top eigen‑values/vectors (only positive ones)
        # ---------------------------------------------------------------------
        eigvals, eigvecs = np.linalg.eigh(A)
        pos = eigvals > 0.0
        eigvals = eigvals[pos]
        eigvecs = eigvecs[:, pos]

        # sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Trim to the requested number of components
        k = min(len(eigvals), n_components)
        sqrt_eig = np.sqrt(eigvals[:k])
        B = eigvecs[:, :k] * sqrt_eig  # shape (n, k)

        # ---------------------------------------------------------------------
        # 3) CVXPY optimisation
        # ---------------------------------------------------------------------
        X = cp.Variable((n, k))
        objective = cp.Minimize(
            cp.sum_squares(B - X) + sparsity * cp.norm1(X)
        )
        constraints = [cp.norm(X[:, i]) <= 1.0 for i in range(k)]
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.OSQP, warm_start=True)  # fast convex solver
        except Exception:
            return {"components": [], "explained_variance": []}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or X.value is None:
            return {"components": [], "explained_variance": []}

        # ---------------------------------------------------------------------
        # 4) Extract components and compute explained variance
        # ---------------------------------------------------------------------
        components = X.value  # shape (n, k)
        # Variance explained: vᵀ A v for each component v
        explained_variance = (
            np.einsum("ij,ji->i", components.T, A @ components).tolist()
        )

        return {
            "components": components.tolist(),
            "explained_variance": explained_variance,
        }