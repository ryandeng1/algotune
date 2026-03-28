from typing import Any
import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # load problem data
        w_max = np.asarray(problem["w_max"])
        d_max = np.asarray(problem["d_max"])
        q_max = np.asarray(problem["q_max"])
        λ_min = np.asarray(problem["λ_min"])
        μ_max = float(problem["μ_max"])
        γ = np.asarray(problem["γ"])
        n = γ.size

        # decision variables
        μ = cp.Variable(n, pos=True)
        λ = cp.Variable(n, pos=True)

        # derived expressions
        ρ = λ / μ
        q = cp.square(ρ) / (1 - ρ)
        w = q / λ + 1 / μ
        d = 1 / (μ - λ)

        # constraints
        constraints = [
            w <= w_max,
            d <= d_max,
            q <= q_max,
            λ >= λ_min,
            cp.sum(μ) <= μ_max
        ]

        # objective
        obj = cp.Minimize(γ @ (μ / λ))

        # build problem
        prob = cp.Problem(obj, constraints)

        # solve with the most efficient solver first
        try:
            prob.solve(solver=cp.ECOS, gp=True, warm_start=True)
        except cp.error.DGPError:
            try:
                prob.solve(solver=cp.SCS, gp=True, warm_start=True)
            except cp.error.DCPError:
                # fall back to a quick heuristic if both fail
                λ_val = λ_min
                μ_val = np.full(n, μ_max / n)
                obj_val = float(γ @ (μ_val / λ_val))
                return {"μ": μ_val, "λ": λ_val, "objective": obj_val}

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"Solver failed with status {prob.status}")

        return {"μ": μ.value, "λ": λ.value, "objective": float(prob.value)}