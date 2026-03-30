# solver.py
from typing import Any, Dict
import cvxpy as cp
import numpy as np


class Solver:
    """
    Lyapunov stability checker based on semidefinite programming.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the Lyapunov stability analysis problem.

        Parameters
        ----------
        problem : dict
            Expected to contain the key 'A' with a square matrix.

        Returns
        -------
        dict
            {'is_stable': bool, 'P': list or None}

        Notes
        -----
        The solver is kept lightweight: it only builds the
        constraints once and delegates the heavy lifting to CVXPY.
        """
        A = np.asarray(problem["A"], dtype=np.float64)
        n = A.shape[0]

        # Decision variable: symmetric positive definite matrix P
        P = cp.Variable((n, n), symmetric=True)

        # Lyapunov constraints:  P >> I ,  A^T P A - P << -I
        constraints = [
            P >> np.eye(n, dtype=np.float64),
            A.T @ P @ A - P << -np.eye(n, dtype=np.float64),
        ]

        prob = cp.Problem(cp.Minimize(0), constraints)

        try:
            # Use the default solver (SCS) – it is robust and fast for small to medium sized SDP.
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-6)
        except Exception:
            return {"is_stable": False, "P": None}

        if prob.status in {"optimal", "optimal_inaccurate"}:
            return {"is_stable": True, "P": P.value.tolist()}
        return {"is_stable": False, "P": None}