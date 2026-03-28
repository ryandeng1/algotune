import numpy as np
import scipy.optimize

class Solver:
    def __init__(self, a2, a3, a4, a5):
        self.a2, self.a3, self.a4, self.a5 = a2, a3, a4, a5

    @staticmethod
    def func(x, a0, a1, a2, a3, a4, a5):
        return a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4 + a5 * x ** 5

    @staticmethod
    def fprime(x, a0, a1, a2, a3, a4, a5):
        return a1 + 2 * a2 * x + 3 * a3 * x ** 2 + 4 * a4 * x ** 3 + 5 * a5 * x ** 4

    def solve(self, problem: dict[str, list[float]]) -> dict[str, list[float]]:
        # Input vectors
        x0_arr = np.array(problem['x0'], dtype=float)
        a0_arr = np.array(problem['a0'], dtype=float)
        a1_arr = np.array(problem['a1'], dtype=float)

        # Validate lengths
        if x0_arr.size != a0_arr.size or x0_arr.size != a1_arr.size:
            return {'roots': [float('nan')] * x0_arr.size}

        # Fast vectorised Newton
        args = (a0_arr, a1_arr, self.a2, self.a3, self.a4, self.a5)
        try:
            roots = scipy.optimize.newton(
                self.func, x0_arr, fprime=self.fprime, args=args
            )
        except Exception:
            return {'roots': [float('nan')] * x0_arr.size}

        # Ensure output is array, fill missing with NaN if necessary
        if np.isscalar(roots):
            roots = np.full_like(x0_arr, roots, dtype=float)
        else:
            if len(roots) < x0_arr.size:
                roots = np.concatenate(
                    (roots, np.full(x0_arr.size - len(roots), float('nan')))
                )
        return {'roots': roots.tolist()}