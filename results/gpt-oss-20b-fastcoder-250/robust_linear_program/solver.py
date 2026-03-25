# solver.py
from typing import Any
import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves a robust LP formulated as a second-order cone program (SOCP).

        Parameters
        ----------
        problem : dict
            A dictionary containing:
                - "c": list or array of shape (n,)
                - "b": list or array of shape (m,)
                - "P": list of m arrays, each of shape (n, n)
                - "q": list of m arrays, each of shape (n,)

        Returns
        -------
        dict
            A dictionary with keys:
                - "objective_value": float, optimal objective value
                - "x": list of floats, optimal solution vector
        """
        c = np.asarray(problem["c"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)
        P = np.asarray(problem["P"], dtype=float)
        q = np.asarray(problem["q"], dtype=float)

        n = c.shape[0]
        m = b.shape[0]

        x = cp.Variable(n)

        constraints = []
        for i in range(m):
            # SOC constraint: ||P_i^T x||_2 <= b_i - q_i^T x
            constraints.append(cp.SOC(b[i] - q[i].T @ x, P[i].T @ x))

        prob = cp.Problem(cp.Minimize(c @ x), constraints)

        try:
            # Use SCS solver which supports SOC constraints and is generally fast.
            prob.solve(solver=cp.SCS, verbose=False, max_iters=5000, eps=1e-5)
        except Exception:
            return {"objective_value": float("inf"), "x": [float("nan")] * n}

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return {"objective_value": float("inf"), "x": [float("nan")] * n}

        return {"objective_value": float(prob.value), "x": sanArray_to_list(x.value)}

def sanArray_to_list(arr):
    """Convert a NumPy array or CVXPY array to a plain Python list."""
    if arr is None:
        return [float("nan")]
    return arr.tolist()
