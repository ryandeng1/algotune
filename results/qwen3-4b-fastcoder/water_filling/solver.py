from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        alpha = np.asarray(problem["alpha"], dtype=float)
        P_total = float(problem["P_total"])
        n = alpha.size

        if n == 0 or P_total <= 0 or not np.all(alpha > 0):
            return {"x": [float("nan")] * n, "Capacity": float("nan")}

        x_var = cp.Variable(n, nonneg=True)
        objective = cp.Maximize(cp.sum(cp.log(alpha + x_var)))
        constraints = [cp.sum(x_var) == P_total]

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
        except cp.SolverError as e:
            return {"x": [float("nan")] * n, "Capacity": float("nan")}

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or x_var.value is None:
            return {"x": [float("nan")] * n, "Capacity": float("nan")}

        x_val = x_var.value
        current_sum = np.sum(x_val)
        if current_sum > 1e-9:
            x_scaled = x_val * (P_total / current_sum)
        else:
            x_scaled = x_val

        capacity_terms = np.log(alpha + x_scaled)
        final_capacity = np.sum(capacity_terms)

        return {"x": x_scaled.tolist(), "Capacity": float(final_capacity)}