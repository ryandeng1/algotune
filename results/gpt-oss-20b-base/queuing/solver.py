# solver.py
from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    """
    Optimizes the stochastic resource allocation problem defined in the
    `problem` dictionary. The implementation relies solely on cvxpy
    with geometric programming enabled. All problem data are converted
    to cvxpy Constants to avoid unnecessary copies.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert all inputs to numpy arrays (required for broadcasting)
        w_max = np.asarray(problem["w_max"])
        d_max = np.asarray(problem["d_max"])
        q_max = np.asarray(problem["q_max"])
        λ_min = np.asarray(problem["λ_min"])
        μ_max = float(problem["μ_max"])
        γ = np.asarray(problem["γ"])

        n = γ.size
        # CVXPY variables (positive)
        μ = cp.Variable(n, pos=True)
        λ = cp.Variable(n, pos=True)

        # Intermediate expressions
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

        # Objective : minimize sum γ_i * (μ_i / λ_i)
        obj = cp.Minimize(γ @ (μ / λ))

        # Problem declaration
        prob = cp.Problem(obj, constraints)

        # Solve once using the default solver (SCS or ECOS, depending on availability)
        # The following flag instructs CVXPY to treat the problem as a geometric program.
        try:
            prob.solve(gp=True, verbose=False, eps=1e-7)
        except cp.error.DGPError:
            # Fall back to standard DCP if GP interpretation fails
            prob.solve(verbose=False, eps=1e-7)

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"Solver failed with status {prob.status}")

        # Return the optimal values
        return {"μ": μ.value, "λ": λ.value, "objective": float(prob.value)}