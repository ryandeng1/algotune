from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Extract and validate inputs
        alpha = np.asarray(problem["alpha"], dtype=float)
        P_total = float(problem["P_total"])
        n = alpha.size

        if n == 0 or P_total <= 0 or not np.all(alpha > 0):
            return {"x": [float("nan")] * n, "Capacity": float("nan")}

        # Formulate convex optimisation problem
        x_var = cp.Variable(n, nonneg=True)
        objective = cp.Maximize(cp.sum(cp.log(alpha + x_var)))
        constraint = [cp.sum(x_var) == P_total]
        prob = cp.Problem(objective, constraint)

        try:
            prob.solve()
        except cp.SolverError:
            return {"x": [float("nan")] * n, "Capacity": float("nan")}

        # Handle non‑optimal status
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x_var.value is None:
            return {"x": [float("nan")] * n, "Capacity": float("nan")}

        x_val = np.array(x_var.value).flatten()
        reported_capacity = float(prob.value) if prob.value is not None else float("nan")

        # --- Scaling to meet budget constraint exactly ---
        current_sum = np.sum(x_val)
        if current_sum > 1e-12:
            x_scaled = x_val * (P_total / current_sum)
        else:
            x_scaled = x_val.copy()        # keep as is if sum is near zero

        # Remove any tiny negative values introduced by scaling
        x_scaled = np.maximum(x_scaled, 0.0)

        # If clipping changed the sum, re‑scale once more
        final_sum = np.sum(x_scaled)
        if final_sum > 1e-12 and not np.isclose(final_sum, P_total, rtol=1e-9):
            x_scaled *= P_total / final_sum
        # --- End of scaling ---

        # Compute capacity safely
        safe_x = np.maximum(x_scaled, 0)                # defensive
        terms = np.log(alpha + safe_x)
        if np.all(np.isfinite(terms)):
            final_capacity = float(np.sum(terms))
        else:
            final_capacity = reported_capacity          # fallback

        return {"x": x_scaled.tolist(), "Capacity": final_capacity}