from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        P = np.asarray(problem["P"], float)
        q = np.asarray(problem["q"], float)
        G = np.asarray(problem["G"], float)
        h = np.asarray(problem["h"], float)
        A = np.asarray(problem["A"], float)
        b = np.asarray(problem["b"], float)
        n = P.shape[0]

        P = (P + P.T) / 2
        x = cp.Variable(n)
        objective = 0.5 * cp.quad_form(x, P) + q @ x
        constraints = [G @ x <= h, A @ x == b]
        prob = cp.Problem(cp.Minimize(objective), constraints)
        optimal_value = prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8)

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"Solver failed (status = {prob.status})")

        return {"solution": x.value.tolist(), "objective": float(optimal_value)}