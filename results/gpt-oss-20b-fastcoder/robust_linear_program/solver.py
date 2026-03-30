from typing import Any, Dict
import cvxpy as cp
import numpy as np

class Solver:
    """
    Fast solver for the robust linear program

        minimize   cᵀ x
        subject to SOC(b_i - q_iᵀ x, P_iᵀ x),  i = 1,…,m

    where SOC(a, b) denotes the vector second‑order cone: a ≥ ‖b‖₂.

    The formulation is identical to the original implementation but it
    removes all unnecessary statements, avoids explicit loops in Python
    when needed, and uses a compact ``cp.Problem`` construction.
    """

    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Any]:
        # Extract parameters
        c = np.asarray(problem["c"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)
        P = np.asarray(problem["P"], dtype=np.float64)   # shape (m, n, n)
        q = np.asarray(problem["q"], dtype=np.float64)   # shape (m, n)

        n = c.shape[0]
        m = b.size

        # Decision variable
        x = cp.Variable(n)

        # Build SOC constraints
        constraints = []
        for i in range(m):
            constraints.append(
                cp.SOC(b[i] - q[i].T @ x, P[i].T @ x)
            )

        # Problem definition
        objective = cp.Minimize(c.T @ x)
        prob = cp.Problem(objective, constraints)

        # Solve
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            return {"objective_value": np.inf, "x": np.full(n, np.nan)}

        # Return result only if optimal
        if prob.status in {"infeasible", "unbounded"}:
            return {"objective_value": np.inf, "x": np.full(n, np.nan)}
        return {"objective_value": prob.value, "x": x.value}