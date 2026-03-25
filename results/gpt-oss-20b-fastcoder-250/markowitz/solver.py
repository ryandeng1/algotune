# solver.py
import numpy as np
from typing import Any, Dict, List
from scipy.optimize import minimize, LinearConstraint, Bounds


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the Markowitz portfolio optimisation problem:
            maximize μᵀ w - γ wᵀ Σ w
            subject to wᵀ 1 = 1,  w ≥ 0
        using scipy's efficient Trust-Constr optimizer.
        """
        # Convert inputs to numpy arrays
        μ = np.asarray(problem["μ"], dtype=np.float64)
        Σ = np.asarray(problem["Σ"], dtype=np.float64)
        γ = float(problem["γ"])
        n = μ.size

        # Objective: f(w) = 0.5 * wᵀ H w - μᵀ w with H = 2γ Σ, then maximize = minimize negative
        H = 2.0 * γ * Σ
        # We minimize: 0.5 wᵀ H w - μᵀ w
        def obj(w):
            return 0.5 * w @ (H @ w) - μ @ w

        def grad(w):
            return H @ w - μ

        # Constraints: sum(w) = 1
        lin = LinearConstraint(np.ones((1, n)), lb=1.0, ub=1.0)
        # Bounds: w >= 0
        bounds = Bounds(0.0, None)

        # Initial guess: uniform weights
        x0 = np.full(n, 1.0 / n)

        # Use Trust-Constr solver which handles nonlinear objective with bound and linear equality
        res = minimize(
            obj,
            x0,
            method="trust-constr",
            jac=grad,
            constraints=[lin],
            bounds=bounds,
            options={"gtol": 1e-8, "ftol": 1e-8, "xtol": 1e-8, "maxiter": 200, "verbose": 0},
        )

        if not res.success:
            return None

        w_opt = res.x
        # Sanity checks similar to reference implementation
        if not np.isfinite(w_opt).all():
            return None
        if abs(np.sum(w_opt) - 1.0) > 1e-4:
            return None
        if (w_opt < -1e-6).any():
            return None

        return {"w": w_opt.tolist()}
