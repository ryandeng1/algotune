import numpy as np
from typing import Any
import cvxpy as cp

class Solver:
    """
    Very fast solver for a mean‑variance portfolio problem with
    non‑negative weights and unit budget constraint.

    The formulation is

        max   μᵀw – γ * wᵀΣw
        s.t.  1ᵀw = 1
              w ≥ 0

    The problem is strictly convex; we use CVXPY with the efficient
    OSQP solver.  OSQP is a first‑order solver that is extremely fast
    for problems of this size (hundreds or thousands of variables).
    It also returns the optimal solution directly, avoiding the
    overhead of a general purpose solver.

    Because the problem is convex and the constraints are linear,
    the solution found by OSQP is globally optimal.
    """
    def solve(self, problem: dict[str, Any]) -> dict[str, list[float]] | None:
        # Turn inputs into efficient NumPy arrays
        μ = np.asarray(problem["μ"], dtype=np.float64)
        Σ = np.asarray(problem["Σ"], dtype=np.float64)
        γ = float(problem["γ"])

        n = μ.size
        w = cp.Variable(n)

        expr = μ @ w - γ * cp.quad_form(w, cp.psd_wrap(Σ))
        objective = cp.Maximize(expr)

        constraints = [cp.sum(w) == 1.0, w >= 0.0]

        # Solve using OSQP – the fastest generic solver in CVXPY for QPs
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=False, warm_start=True)
        except cp.error.SolverError:
            return None

        # Validate solution
        if w.value is None or not np.all(np.isfinite(w.value)):
            return None

        return {"w": w.value.tolist()}