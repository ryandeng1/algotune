from typing import Any, Dict
import numpy as np
import cvxpy as cp


class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Solves a robust LP using CVXPY with a vectorized Second‑Order Cone formulation.

        Parameters
        ----------
        problem : dict
            Contains the LP data:
            - "c" : ndarray, shape (n,)
            - "b" : ndarray, shape (m,)
            - "P" : ndarray, shape (m, n, n) (symmetric positive semi‑definite)
            - "q" : ndarray, shape (m, n)

        Returns
        -------
        dict
            Contains the optimal objective value and optimal `x`:
            - "objective_value" : float
            - "x" : ndarray, shape (n,)
        """
        # Pull data out of the dictionary once
        c, b, P, q = (
            np.asarray(problem["c"]),
            np.asarray(problem["b"]),
            np.asarray(problem["P"]),
            np.asarray(problem["q"]),
        )
        m, n = c.shape[0], b.shape[0]

        # Decision variable
        x = cp.Variable(n)

        # Build constraints efficiently in one go
        # Expression for each cone: (b_i - q_i^T x, P_i^T x)
        # We construct vectors of the first and second arguments
        # then stack them into an array shape (m, n+1) for cp.SOC
        first = b - cp.multiply(q, x) @ np.ones(n)
        second = cp.vstack([P[i].T @ x for i in range(m)])
        constraints = [cp.SOC(first[i], second[i]) for i in range(m)]

        prob = cp.Problem(cp.Minimize(c @ x), constraints)

        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
            if prob.status not in {"optimal", "optimal_inaccurate"}:
                return {"objective_value": np.inf, "x": np.full(n, np.nan)}
            return {"objective_value": prob.value, "x": x.value}
        except Exception:
            return {"objective_value": np.inf, "x": np.full(n, np.nan)}