import numpy as np
import cvxpy as cp

def solve(problem: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # Convert inputs to NumPy arrays
    w_max = np.asarray(problem["w_max"])
    d_max = np.asarray(problem["d_max"])
    q_max = np.asarray(problem["q_max"])
    λ_min = np.asarray(problem["λ_min"])
    μ_max = float(problem["μ_max"])
    γ = np.asarray(problem["γ"])

    n = γ.size
    # Decision variables
    μ = cp.Variable(n, pos=True)
    λ = cp.Variable(n, pos=True)

    # Derived expressions
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

    # Objective: minimize weighted sum of μ/λ
    obj = cp.Minimize(γ @ (μ / λ))

    # Problem definition
    prob = cp.Problem(obj, constraints)

    # Solve – try geometric programming first, then fallback
    try:
        prob.solve(gp=True, verbose=False, max_iters=5000)
    except cp.error.DGPError:
        prob.solve(verbose=False, max_iters=5000)

    # Verify solution status
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise ValueError(f"Solver failed with status {prob.status}")

    return {"μ": μ.value, "λ": λ.value, "objective": float(prob.value)}