import numpy as np
from scipy.optimize import minimize

def solve(problem: dict) -> dict[str, list[float]] | None:
    mu = np.asarray(problem["μ"], dtype=float)
    Sigma = np.asarray(problem["Σ"], dtype=float)
    gamma = float(problem["γ"])
    n = mu.size

    def objective(w):
        return - (mu @ w - gamma * 0.5 * np.dot(w, Sigma @ w))

    constraints = (
        # sum(w) == 1
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    )
    bounds = [(0, None)] * n

    res = minimize(objective, np.full(n, 1.0 / n), method="SLSQP",
                   constraints=constraints, bounds=bounds,
                   options={"ftol": 1e-12, "gtol": 1e-12, "maxiter": 1000})

    if not res.success or not np.isfinite(res.x).all():
        return None
    return {"w": res.x.tolist()}