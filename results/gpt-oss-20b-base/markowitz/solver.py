# solver.py
from typing import Any, Dict, List
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Efficiently solve the Markowitz portfolio optimisation:
            maximise   μ^T w - γ w^T Σ w
            s.t.       1^T w = 1
                       w >= 0
        Using SciPy's SLSQP optimizer which is fast for this small constrained
        quadratic programming task.
        """
        μ = np.asarray(problem["μ"], dtype=np.float64)
        Σ = np.asarray(problem["Σ"], dtype=np.float64)
        γ = float(problem["γ"])
        n = μ.size

        # Objective: - (μ^T w - γ w^T Σ w)
        def obj(w: np.ndarray) -> float:
            return - (μ @ w - γ * w @ Σ @ w)

        # Gradient of the objective
        def grad(w: np.ndarray) -> np.ndarray:
            return - (μ - 2 * γ * Σ @ w)

        # Equality constraint: sum(w) = 1
        cons = LinearConstraint(np.ones((1, n)), [1.0], [1.0])

        # Bounds: w_i >= 0
        bounds = Bounds(0.0, np.inf)

        # Initial guess: equal weights
        x0 = np.full(n, 1.0 / n)

        res = minimize(
            obj,
            x0,
            method="SLSQP",
            jac=grad,
            constraints=[cons],
            bounds=bounds,
            options={"ftol": 1e-12, "disp": False, "maxiter": 2000},
        )

        if not res.success:
            # Fallback: return None if optimisation fails
            return None

        w = res.x
        # Ensure numerical stability: clip tiny negatives to zero
        w = np.where(w < 0.0, 0.0, w)
        # Re-normalise to sum to 1 due to clipping
        w = w / w.sum() if w.sum() > 0 else w

        return {"w": w.tolist()}
