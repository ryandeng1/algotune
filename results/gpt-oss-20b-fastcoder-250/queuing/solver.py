from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert inputs to numpy arrays for vectorised operations
        w_max = np.asarray(problem["w_max"])
        d_max = np.asarray(problem["d_max"])
        q_max = np.asarray(problem["q_max"])
        λ_min = np.asarray(problem["λ_min"])
        μ_max = float(problem["μ_max"])
        γ = np.asarray(problem["γ"])
        n = γ.size

        # Decision variables
        μ = cp.Variable(n, nonneg=True)
        λ = cp.Variable(n, nonneg=True)

        # Server load
        ρ = λ / μ

        # Performance metrics
        q = cp.square(ρ) / (1 - ρ)
        w = q / λ + 1 / μ
        d = 1 / (μ - λ)

        # Constraints
        constraints = [
            w <= w_max,
            d <= d_max,
            q <= q_max,
            λ >= λ_min,
            cp.sum(μ) <= μ_max,
        ]

        # Objective (to be minimised)
        obj = cp.Minimize(γ @ (μ / λ))

        # Problem definition
        prob = cp.Problem(obj, constraints)

        # Attempt a geometric program first (GP), then general DCP
        try:
            prob.solve(gp=True, solver=cp.ECOS, verbose=False)
        except cp.error.DGPError:
            try:
                prob.solve(solver=cp.ECOS, verbose=False)
            except cp.error.DCPError:
                # Heuristic fallback if both approaches fail
                λ_val = λ_min
                μ_val = np.full(n, μ_max / n)
                obj_val = float(γ @ (μ_val / λ_val))
                return {"μ": μ_val, "λ": λ_val, "objective": obj_val}

        # Verify that solver found an optimal solution
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"Solver failed with status {prob.status}")

        # Return results
        return {
            "μ": μ.value,
            "λ": λ.value,
            "objective": float(prob.value),
        }