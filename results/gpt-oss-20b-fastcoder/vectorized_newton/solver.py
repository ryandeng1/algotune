import numpy as np
import scipy.optimize

# Assuming these are defined elsewhere:
# _task_f_vec and _task_f_vec_prime

class Solver:
    def __init__(self):
        self.a2 = 1e-9
        self.a3 = 0.004
        self.a4 = 10.0
        self.a5 = 0.27456
        self.fprime = _task_f_vec_prime
        self.func = _task_f_vec

    def solve(self, problem: dict[str, list[float]]) -> dict[str, list[float]]:
        # Convert inputs to numpy arrays
        x0 = np.asanyarray(problem.get("x0", []), dtype=float)
        a0 = np.asanyarray(problem.get("a0", []), dtype=float)
        a1 = np.asanyarray(problem.get("a1", []), dtype=float)

        n = x0.size
        if a0.size != n or a1.size != n:
            return {"roots": [float("nan")] * n}

        args = (a0, a1, self.a2, self.a3, self.a4, self.a5)

        try:
            roots = scipy.optimize.newton(
                self.func, x0, fprime=self.fprime, args=args
            )
            if np.isscalar(roots):
                roots = np.array([roots], dtype=float)
            else:
                roots = np.asarray(roots, dtype=float)
            if roots.size != n:
                # Pad or truncate to the expected length
                if roots.size < n:
                    roots = np.concatenate([roots, np.full(n - roots.size, np.nan)])
                else:
                    roots = roots[:n]
        except Exception:
            roots = np.full(n, np.nan, dtype=float)

        return {"roots": roots.tolist()}