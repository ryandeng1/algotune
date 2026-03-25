# solver.py
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import cvxpy as cp

class Solver:
    """
    Solver for the M/M/1 queue optimization problem.
    Uses CVXPY with a geometric program formulation for speed.
    """
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        w_max = np.asarray(problem["w_max"])
        d_max = np.asarray(problem["d_max"])
        q_max = np.asarray(problem["q_max"])
        λ_min = np.asarray(problem["λ_min"])
        μ_max = float(problem["μ_max"])
        γ = np.asarray(problem["γ"])
        n = γ.size

        # CVXPY variables
        μ = cp.Variable(n, pos=True)
        λ = cp.Variable(n, pos=True)

        ρ = λ / μ  # server load ρ in (0,1)

        # queue‑length, waiting time, total delay
        q = cp.square(ρ) / (1 - ρ)
        w = q / λ + 1 / μ
        d = 1 / (μ - λ)

        constraints = [
            w <= w_max,
            d <= d_max,
            q <= q_max,
            λ >= λ_min,
            cp.sum(μ) <= μ_max,
        ]

        objective = cp.Minimize(γ @ (μ / λ))
        prob = cp.Problem(objective, constraints)

        # Solve as a geometric program (most efficient for this problem)
        try:
            prob.solve(gp=True, solver=cp.ECOS, verbose=False)
        except cp.error.DGPError:
            # fallback to standard solver if GP fails
            prob.solve(solver=cp.ECOS, verbose=False)

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"Solver failed with status {prob.status}")

        return {
            "μ": μ.value,
            "λ": λ.value,
            "objective": float(prob.value),
        }
