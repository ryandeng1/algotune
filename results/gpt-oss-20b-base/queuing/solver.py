import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        w_max = np.asarray(problem["w_max"])
        d_max = np.asarray(problem["d_max"])
        q_max = np.asarray(problem["q_max"])
        λ_min = np.asarray(problem["λ_min"])
        μ_max = float(problem["μ_max"])
        γ = np.asarray(problem["γ"])

        n = γ.size

        # Simple heuristic: distribute μ evenly and set λ to λ_min
        μ = np.full(n, μ_max / n)
        λ = λ_min.copy()

        # Clip λ to ensure feasibility
        λ = np.minimum(λ, μ * (1 - 1e-6))  # keep ρ < 1

        # Compute objective
        obj = float(np.dot(γ, μ / λ))

        # Return values
        return {"μ": μ, "λ": λ, "objective": obj}