from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        w_max = np.asarray(problem["w_max"])
        d_max = np.asarray(problem["d_max"])
        q_max = np.asarray(problem["q_max"])
        l_min = np.asarray(problem["λ_min"])
        m_max = float(problem["μ_max"])
        gamma = np.asarray(problem["γ"])
        n = gamma.size

        m = cp.Variable(n, pos=True)
        l = cp.Variable(n, pos=True)
        r = l / m

        q = cp.power(r, 2) / (1 - r)
        w = q / l + 1 / m
        d = 1 / (m - l)

        constraints = [
            w <= w_max,
            d <= d_max,
            q <= q_max,
            l >= l_min,
            cp.sum(m) <= m_max,
        ]
        obj = cp.Minimize(gamma @ (m / l))
        prob = cp.Problem(obj, constraints)

        try:
            prob.solve(gp=True)
        except cp.error.DGPError:
            try:
                prob.solve()
            except cp.error.DCPError:
                l_val = l_min
                m_val = np.full(n, m_max / n)
                obj_val = float(gamma @ (m_val / l_val))
                return {"μ": m_val, "λ": l_val, "objective": obj_val}

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"Solver failed with status {prob.status}")

        return {
            "μ": m.value,
            "λ": l.value,
            "objective": float(prob.value),
        }