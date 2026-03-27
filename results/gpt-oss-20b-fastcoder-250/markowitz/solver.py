import numpy as np
from typing import Any, Dict, List, Union
import cvxpy as cp

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Union[Dict[str, List[float]], None]:
        # Extract data
        μ = np.asarray(problem["μ"], dtype=np.float64)
        Σ = np.asarray(problem["Σ"], dtype=np.float64)
        γ = float(problem["γ"])
        n = μ.size

        # Decision variable
        w = cp.Variable(n)

        # Objective: maximize μ'w - γ w^T Σ w
        # Quad form with quadratic term (positive‑semidefinite)
        obj = cp.Maximize(μ @ w - γ * cp.quad_form(w, Σ))

        # Constraints: weights sum to 1 and are non‑negative
        constraints = [cp.sum(w) == 1, w >= 0]

        # Solve, prefer the fastest convex quadratic solver that handles
        # bound constraints efficiently.
        try:
            prob = cp.Problem(obj, constraints)
            # OSQP works well for quadratic problems with simple bounds.
            prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-8, eps_rel=1e-8)
        except cp.error.SolverError:
            return None

        # Check feasibility and finite solution
        if w.value is None or not np.isfinite(w.value).all():
            return None

        return {"w": w.value.tolist()}