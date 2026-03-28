from typing import Any
import numpy as np
import scipy.optimize

class Solver:
    def solve(self, problem: dict[str, list[float]]) -> dict[str, list[float]]:
        try:
            x0_arr = np.array(problem["x0"])
            a0_arr = np.array(problem["a0"])
            a1_arr = np.array(problem["a1"])
            n = len(x0_arr)
            if len(a0_arr) != n or len(a1_arr) != n:
                raise ValueError("Input arrays have mismatched lengths")
        except Exception as e:
            return {"roots": []}

        args = (a0_arr, a1_arr, self.a2, self.a3, self.a4, self.a5)

        try:
            roots_arr = scipy.optimize.newton(self.func, x0_arr, fprime=self.fprime, args=args)
        except RuntimeError as e:
            return {"roots": [float("nan")] * n}

        if np.isscalar(roots_arr):
            roots_arr = np.array([roots_arr])

        roots_arr = np.pad(roots_arr, (0, n - len(roots_arr)), 'constant', constant_values=np.nan)
        roots_arr = roots_arr[:n]

        return {"roots": roots_arr.tolist()}