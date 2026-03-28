import numpy as np
import cvxpy as cp

def solve(problem: dict[str, int | float | list[float]]) -> dict[str, list[float]] | None:
    """
    Mean‑and‑variance portfolio optimisation

    Maximise  μᵀw - γ·wᵀΣw
    subject to  sum(w) == 1,  w >= 0
    """
    μ = np.asarray(problem["μ"], dtype=float)
    Σ = np.asarray(problem["Σ"], dtype=float)
    γ = float(problem["γ"])

    n = μ.size
    w = cp.Variable(n)

    # Objective function
    objective = cp.Maximize(μ @ w - γ * cp.quad_form(w, Σ))

    # Constraints
    constraints = [cp.sum(w) == 1, w >= 0]

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP, warm_start=True, refresh=False)
    except cp.error.SolverError:
        return None

    # Validate the solution
    if w.value is None or not np.isfinite(w.value).all():
        return None

    return {"w": w.value.tolist()}