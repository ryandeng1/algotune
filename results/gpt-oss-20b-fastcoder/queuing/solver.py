from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert problem data to numpy arrays / scalars
        w_max = np.asarray(problem["w_max"], float)
        d_max = np.asarray(problem["d_max"], float)
        q_max = np.asarray(problem["q_max"], float)
        λ_min = np.asarray(problem["λ_min"], float)
        μ_max = float(problem["μ_max"])
        γ = np.asarray(problem["γ"], float)

        n = γ.size

        # Decision variables
        μ = cp.Variable(n, pos=True)
        λ = cp.Variable(n, pos=True)

        ρ = λ / μ
        q = cp.power(ρ, 2) / (1 - ρ)
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

        # Objective
        obj = cp.Minimize(γ @ (μ / λ))

        # Problem
        prob = cp.Problem(obj, constraints)

        # Solve – try geometric programming first, fallback to standard solvers
        try:
            prob.solve(gp=True)
        except cp.error.DGPError:
            prob.solve()

        # Check feasibility
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"Solver failed with status {prob.status}")

        return {"μ": μ.value, "λ": λ.value, "objective": float(prob.value)}