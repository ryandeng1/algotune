from typing import Any
import numpy as np
import cvxpy as cp

def solve(problem: dict[str, Any]) -> dict[str, Any]:
    # Extract problem data
    w_max = np.asarray(problem["w_max"], dtype=float)
    d_max = np.asarray(problem["d_max"], dtype=float)
    q_max = np.asarray(problem["q_max"], dtype=float)
    λ_min = np.asarray(problem["λ_min"], dtype=float)
    μ_max = float(problem["μ_max"])
    γ = np.asarray(problem["γ"], dtype=float)

    n = γ.size

    # Decision variables
    μ = cp.Variable(n, pos=True)
    λ = cp.Variable(n, pos=True)

    # Expressions
    ρ = λ / μ
    q = cp.power(ρ, 2) / (1 - ρ)
    w = q / λ + 1 / μ
    d = 1 / (μ - λ)

    # Constraints
    constraints = [w <= w_max,
                   d <= d_max,
                   q <= q_max,
                   λ >= λ_min,
                   cp.sum(μ) <= μ_max]

    # Objective
    obj = cp.Minimize(γ @ (μ / λ))

    # Problem
    prob = cp.Problem(obj, constraints)

    # Solve
    prob.solve(gp=True, eps=1e-6, verbose=False)

    # Fallback if GP fails
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        # Try ordinary solve
        try:
            prob.solve(verbose=False)
        except cp.error.DCPError:
            # Default feasible point
            μ_val = np.full(n, μ_max / n)
            λ_val = λ_min
            obj_val = float(γ @ (μ_val / λ_val))
            return {"μ": μ_val, "λ": λ_val, "objective": obj_val}

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise ValueError(f"Solver failed with status {prob.status}")

    return {"μ": μ.value, "λ": λ.value, "objective": float(prob.value)}