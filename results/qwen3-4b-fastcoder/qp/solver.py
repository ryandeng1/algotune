from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        P = np.array(problem["P"], dtype=float)
        q = np.array(problem["q"], dtype=float)
        G = np.array(problem["G"], dtype=float)
        h = np.array(problem["h"], dtype=float)
        A = np.array(problem["A"], dtype=float)
        b = np.array(problem["b"], dtype=float)
        n = P.shape[0]

        P = (P + P.T) / 2
        x = cp.Variable(n)
        objective = 0.5 * cp.quad_form(x, cp.psd_wrap(P)) + q @ x
        constraints = [G @ x <= h, A @ x == b]
        prob = cp.Problem(cp.Minimize(objective), constraints)
        optimal_value = prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8)

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"Solver failed (status = {prob.status})")

        return {"solution": x.value.tolist(), "objective": float(optimal_value)}