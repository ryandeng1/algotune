from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    """
    Optimizes a mean‑variance portfolio with non‑negativity and budget constraints.

    The problem is
        max  μᵀw – γ wᵀΣw
        s.t. 1ᵀw = 1
              w ≥ 0

    Parameters
    ----------
    problem : dict
        Dictionary containing ``"μ"``, ``"Σ"``, and ``"γ"``.

    Returns
    -------
    dict
        Optimal portfolio weights `w` in a list form, or ``None`` on failure.
    """
    def solve(self, problem: dict[str, Any]) -> dict[str, list[float]] | None:
        # ------------- Pre‑process data ------------------------------------
        mu = np.asarray(problem["μ"], dtype=np.float64, order="C")
        sigma = np.asarray(problem["Σ"], dtype=np.float64, order="C")
        gamma = float(problem["γ"])

        n = mu.size

        # ------------- Formulate the CVXPY problem -----------------------
        w = cp.Variable(n, nonneg=True)
        # use psd_wrap to avoid enforcing full PSD (quick reduction)
        objective = cp.Maximize(mu @ w - gamma * cp.quad_form(w, cp.psd_wrap(sigma)))
        constraints = [cp.sum(w) == 1]

        prob = cp.Problem(objective, constraints)

        # ------------- Solve -----------------------------------------------
        try:
            # OSQP is the fastest solver for quadratic constraints with
            # inequality constraints – it is deterministic and fast on large
            # industrial‑size data.  If not installed, fall back to the default.
            prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
        except cp.error.SolverError:
            return None

        # ------------- Validation ------------------------------------------
        w_val = w.value
        if w_val is None or not np.isfinite(w_val).all():
            return None

        return {"w": w_val.tolist()}