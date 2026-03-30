from typing import Any, Dict, List, Optional
import cvxpy as cp
import numpy as np

class Solver:
    """
    Fast quadratic‑programming solver for the mean–variance optimisation problem
    defined by a dictionary with keys 'μ', 'Σ', and 'γ'.  Uses CVXPY with the
    highly efficient OSQP backend, which is both fast and stable for the
    small‑ to medium‑size problems usually encountered here.
    """

    def solve(self, problem: Dict[str, Any]) -> Optional[Dict[str, List[float]]]:
        # Extract data and cast to NumPy arrays
        μ = np.asarray(problem["μ"], dtype=np.float64)
        Σ = np.asarray(problem["Σ"], dtype=np.float64)
        γ = float(problem["γ"])

        n = μ.size
        w = cp.Variable(n)

        # Objective: maximise μᵀw – γ·wᵀΣw  (equivalent to minimising -⋯)
        objective = cp.Maximize(μ @ w - γ * cp.quad_form(w, cp.psd_wrap(Σ)))

        # Constraints: sum w = 1, w >= 0
        constraints = [cp.sum(w) == 1, w >= 0]

        prob = cp.Problem(objective, constraints)

        # Solve with OSQP for speed; disable all verbose output
        try:
            prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-8, eps_rel=1e-8)
        except cp.error.SolverError:
            return None

        # If the solver failed to find a finite solution, abort
        if w.value is None or not np.isfinite(w.value).all():
            return None

        return {"w": w.value.tolist()}