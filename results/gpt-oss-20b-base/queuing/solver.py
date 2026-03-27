from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert inputs to NumPy arrays of doubles
        w_max = np.asarray(problem["w_max"], dtype=np.float64)
        d_max = np.asarray(problem["d_max"], dtype=np.float64)
        q_max = np.asarray(problem["q_max"], dtype=np.float64)
        λ_min = np.asarray(problem["λ_min"], dtype=np.float64)
        μ_max = float(problem["μ_max"])
        γ = np.asarray(problem["γ"], dtype=np.float64)

        n = γ.size

        # Decision variables
        μ = cp.Variable(n, pos=True)
        λ = cp.Variable(n, pos=True, name="lambda")
        ρ = λ / μ  # server load

        # M/M/1 formulae
        q = cp.square(ρ) / (1 - ρ)
        w = q / λ + 1 / μ
        d = 1 / (μ - λ)

        # Constraints set
        constraints = [
            w <= w_max,
            d <= d_max,
            q <= q_max,
            λ >= λ_min,
            cp.sum(μ) <= μ_max,
        ]

        # Objective: weighted sum of momentary loads
        obj = cp.Minimize(γ @ (μ / λ))

        # Problem definition
        prob = cp.Problem(obj, constraints)

        # --- solve ---
        # 1) try geometric programming first (fast on convex GP)
        try:
            prob.solve(gp=True, solver=cp.SCS, verbose=False,
                       max_iters=2000, eps=1e-4)
        except cp.error.DGPError:
            # 2) fall back to canonical convex (DCP) solver
            try:
                prob.solve(solver=cp.SCS, verbose=False,
                           max_iters=2000, eps=1e-4)
            except cp.error.DCPError:
                # 3) heuristic fallback: λ = λ_min, μ = μ_max/n
                λ_val = λ_min
                μ_val = np.full(n, μ_max / n, dtype=np.float64)
                obj_val = float(γ @ (μ_val / λ_val))
                return {"μ": μ_val, "λ": λ_val, "objective": obj_val}

        # Validation
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise ValueError(f"Solver failed with status {prob.status}")

        # Return optimized decision variables
        return {
            "μ": μ.value,
            "λ": λ.value,
            "objective": float(prob.value),
        }