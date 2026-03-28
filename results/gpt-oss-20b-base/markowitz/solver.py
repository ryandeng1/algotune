import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem):
        μ = np.asarray(problem['μ'], dtype=float)
        Σ = np.asarray(problem['Σ'], dtype=float)
        γ = float(problem['γ'])
        n = μ.size

        w = cp.Variable(n)
        objective = cp.Maximize(μ @ w - γ * cp.quad_form(w, cp.psd_wrap(Σ)))
        constraints = [cp.sum(w) == 1, w >= 0]

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, max_iter=5000)
        except cp.SolverError:
            return None

        if w.value is None or not np.isfinite(w.value).all():
            return None

        return {'w': w.value.tolist()}