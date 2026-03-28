import numpy as np
import cvxpy as cp
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert all data to contiguous float arrays once
        P = np.asarray(problem['P'], dtype=float)
        q = np.asarray(problem['q'], dtype=float)
        G = np.asarray(problem['G'], dtype=float)
        h = np.asarray(problem['h'], dtype=float)
        A = np.asarray(problem['A'], dtype=float)
        b = np.asarray(problem['b'], dtype=float)

        n = P.shape[0]
        # Make P symmetric to avoid numerical issues
        P = (P + P.T) * 0.5

        # Variable declaration
        x = cp.Variable(n)

        # Objective: 0.5*x.T*P*x + q.T*x
        objective = 0.5 * cp.quad_form(x, cp.psd_wrap(P)) + q @ x

        # Constraints
        constraints = [G @ x <= h, A @ x == b]

        # Solve with OSQP (sparse QP solver)
        prob = cp.Problem(cp.Minimize(objective), constraints)
        optimal_value = prob.solve(
            solver=cp.OSQP,
            eps_abs=1e-8,
            eps_rel=1e-8,
            verbose=False,
        )

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"Solver failed (status={prob.status})")

        return {"solution": x.value.tolist(), "objective": float(optimal_value)}