import numpy as np
from scipy.optimize import minimize

class Solver:
    def solve(self, problem: dict[str, any]) -> dict[str, list[float]] | None:
        mu = np.asarray(problem['μ'], dtype=float)
        cov = np.asarray(problem['Σ'], dtype=float)
        gamma = float(problem['γ'])
        n = mu.size

        # objective : 0.5 w^T G w + c^T w  (here G = 2*γ*cov)
        G = 2.0 * gamma * cov
        c = -mu

        def func(w):
            return 0.5 * w @ G @ w + c @ w

        def grad(w):
            return G @ w + c

        cons = (
            {'type': 'eq', 'fun': lambda w: w.sum() - 1},
            {'type': 'ineq', 'fun': lambda w: w}   # w >= 0
        )
        bounds = [(0.0, None)] * n

        x0 = np.full(n, 1.0 / n)
        res = minimize(func, x0, method='SLSQP', jac=grad,
                       bounds=bounds, constraints=cons,
                       options={'ftol': 1e-12, 'maxiter': 1000})

        if not res.success or not np.isfinite(res.x).all():
            return None
        return {'w': res.x.tolist()}