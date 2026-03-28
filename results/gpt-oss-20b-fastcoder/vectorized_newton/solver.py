import numpy as np
import scipy.optimize

class Solver:
    def __init__(self):
        # placeholder when instance attributes are needed
        self.a2 = self.a3 = self.a4 = self.a5 = 0

    def func(self, x, a0, a1, a2, a3, a4, a5):
        # Example polynomial: a0 + a1*x + a2*x² + a3*x³ + a4*x⁴ + a5*x⁵
        return (((((a5 * x + a4) * x + a3) * x + a2) * x + a1) * x + a0)

    def fprime(self, x, a0, a1, a2, a3, a4, a5):
        # derivative of the above polynomial
        return ((((5 * a5) * x + 4 * a4) * x + 3 * a3) * x + 2 * a2) * x + a1

    def solve(self, problem: dict[str, list[float]]) -> dict[str, list[float]]:
        try:
            x0 = np.asarray(problem['x0'], dtype=float)
            a0 = np.asarray(problem['a0'], dtype=float)
            a1 = np.asarray(problem['a1'], dtype=float)
            a2 = np.asarray(problem.get('a2', 0), dtype=float)
            a3 = np.asarray(problem.get('a3', 0), dtype=float)
            a4 = np.asarray(problem.get('a4', 0), dtype=float)
            a5 = np.asarray(problem.get('a5', 0), dtype=float)
            n = x0.size
            if not (a0.size == a1.size == n):
                raise ValueError()
        except Exception:
            return {'roots': []}

        args = (a0, a1, a2, a3, a4, a5)
        try:
            roots = scipy.optimize.newton(
                self.func,
                x0,
                fprime=self.fprime,
                args=args,
                tol=1e-12,
                maxiter=100
            )
            if np.isscalar(roots):
                roots = np.array([roots])
            else:
                roots = np.asarray(roots)
            if roots.size != n:
                roots = np.concatenate([roots, np.full(n - roots.size, np.nan)])
        except Exception:
            roots = np.full(n, np.nan)

        return {'roots': roots.tolist()}