# solver.py
from typing import Any, Dict
import cvxpy as cp
import numpy as np

class Solver:
    """Fast solver for the CVaR projection problem."""

    def __init__(self) -> None:
        # defaults used by the original implementation
        self.beta = 0.95
        self.kappa = 0.1

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Project x0 onto the set {x | sum_largest(Ax, k) <= alpha}
        where k = floor((1‑beta)*n_scenarios), alpha = kappa*k.

        Parameters
        ----------
        problem : dict
            Must contain:
            - 'x0' : array‑like, shape (n_dims,)
            - 'loss_scenarios' : array‑like, shape (n_scenarios, n_dims)
            May optionally contain 'beta' and 'kappa'.

        Returns
        -------
        dict
            {'x_proj': list of projected coordinates} or an empty list if
            the problem failed to solve.
        """
        # Convert inputs to NumPy arrays once
        x0 = np.asarray(problem['x0'], dtype=np.float64)
        A = np.asarray(problem['loss_scenarios'], dtype=np.float64)

        beta = float(problem.get('beta', self.beta))
        kappa = float(problem.get('kappa', self.kappa))

        n_scenarios, n_dims = A.shape
        # Compute k and alpha
        k = int(np.floor((1 - beta) * n_scenarios))
        alpha = kappa * k

        # Problem definition
        x = cp.Variable(n_dims)
        obj = cp.Minimize(cp.sum_squares(x - x0))
        con = [cp.sum_largest(A @ x, k) <= alpha]
        prob = cp.Problem(obj, con)

        # Solve with a fast solver (SCS is the fastest for dense problems here)
        try:
            prob.solve(solver=cp.SCS, verbose=False,
                       eps=1e-6, max_iters=2000)
        except Exception:
            return {'x_proj': []}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {'x_proj': []}

        return {'x_proj': x.value.tolist()}