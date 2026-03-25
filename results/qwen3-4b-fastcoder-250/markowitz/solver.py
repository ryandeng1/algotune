import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[float]]:
        try:
            μ = np.asarray(problem["μ"], dtype=float)
            Σ = np.asarray(problem["Σ"], dtype=float)
            γ = float(problem["γ"])
            n = μ.size

            w = cp.Variable(n)
            obj = cp.Maximize(μ @ w - γ * cp.quad_form(w, Σ))
            cons = [cp.sum(w) == 1, w >= 0]
            problem = cp.Problem(obj, cons)
            problem.solve(solver=cp.SCS)
        except cp.error.SolverError as e:
            return None

        if w.value is None or not np.isfinite(w.value).all():
            return None

        return {"w": w.value.tolist()}
