from typing import Any, Dict
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Solve a robust linear program with conic (SOC) constraints
        using CVXPY.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - c : (n,) array, coefficients of the linear objective
                - b : (m,) array, RHS scalars of the SOC constraints
                - P : (m, n, n) array, symmetric psd matrices of the SOC constraints
                - q : (m, n) array, linear terms of the SOC constraints

        Returns
        -------
        dict
            Dictionary with keys:
                - objective_value : optimal objective value (float)
                - x : optimal decision variables (array of shape (n,))
        """
        c = np.asarray(problem["c"])
        b = np.asarray(problem["b"])
        P = np.asarray(problem["P"])
        q = np.asarray(problem["q"])

        n, m = c.shape[0], P.shape[0]
        x = cp.Variable(n)

        # Heavy‑lifter: build all SOC constraints in a vectorised way
        socs = [
            cp.SOC(b[i] - q[i] @ x, P[i] @ x)  # the matrix is symmetric, transposition is unnecessary
            for i in range(m)
        ]

        # Form the problem once and solve
        prob = cp.Problem(cp.Minimize(c @ x), socs)

        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
            if prob.status in {"optimal", "optimal_inaccurate"}:
                return {"objective_value": prob.value, "x": x.value}
            # Any other status: treat as infeasible / unbounded
            return {"objective_value": float("inf"), "x": np.full(n, np.nan)}
        except Exception:
            # Solver crashed or failed numerically
            return {"objective_value": float("inf"), "x": np.full(n, np.nan)}