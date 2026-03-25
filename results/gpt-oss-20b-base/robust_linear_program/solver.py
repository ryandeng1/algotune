# solver.py
from typing import Any, Dict, List
import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Solve the robust linear program formulated as a second-order cone program (SOCP).
        Parameters
        ----------
        problem : dict
            Dictionary containing the problem data:
            - "c": list or array of shape (n,)
            - "b": list or array of shape (m,)
            - "P": list of m matrices of shape (n, n)
            - "q": list of m vectors of shape (n,)
        Returns
        -------
        dict
            Contains:
            - "objective_value": optimal objective value (float)
            - "x": optimal solution vector (list of floats)
        """
        # Convert to numpy arrays
        c = np.asarray(problem["c"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)
        P = np.asarray(problem["P"], dtype=np.float64)
        q = np.asarray(problem["q"], dtype=np.float64)

        n = c.size
        m = b.size

        # Define variable
        x = cp.Variable(n)

        # Build SOC constraints: ||P_i.T @ x||_2 <= b_i - q_i^T @ x
        constraints = []
        for i in range(m):
            constraints.append(cp.SOC(b[i] - q[i].T @ x, P[i].T @ x))

        # Define problem
        obj = cp.Minimize(c.T @ x)
        prob = cp.Problem(obj, constraints)

        # Solve with high‑speed cone solver
        try:
            # Use ECOS as default; adjust tolerances for speed/accuracy trade‑off
            prob.solve(solver=cp.ECOS, warm_start=True,
                       verbose=False, eps_abs=1e-6, eps_rel=1e-6,
                       max_iters=1000)
        except Exception:
            return {"objective_value": float("inf"), "x": [float("nan")] * n}

        # Check feasibility of solution
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return {"objective_value": float("inf"), "x": [float("nan")] * n}

        x_val = x.value
        if x_val is None:
            return {"objective_value": float("inf"), "x": [float("nan")] * n}

        return {"objective_value": float(prob.value), "x": x_val.tolist()}
