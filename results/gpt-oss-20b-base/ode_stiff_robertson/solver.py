from typing import Any, List, Dict
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            # return the last state as a plain list of floats
            return {'x0': sol.y[0, -1], 'x1': sol.y[1, -1], 'x2': sol.y[2, -1]}
        else:
            raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: Dict[str, Any], debug: bool = True) -> Any:
        y0 = np.array(problem['y0'], dtype=float, copy=False)
        t0, t1 = float(problem['t0']), float(problem['t1'])
        k_vals = tuple(float(k) for k in problem['k'])
        k1, k2, k3 = k_vals

        # Simple explicit RK4 integrator (1000 steps)
        N = 1000
        h = (t1 - t0) / N

        y = y0.copy()

        def f(t, y):
            y1, y2, y3 = y
            return (
                -k1 * y1 + k3 * y2 * y3,
                k1 * y1 - k2 * y2 ** 2 - k3 * y2 * y3,
                k2 * y2 ** 2,
            )

        for _ in range(N):
            k1_vec = f(t0, y)
            k2_vec = f(t0 + 0.5 * h, y + 0.5 * h * np.array(k1_vec))
            k3_vec = f(t0 + 0.5 * h, y + 0.5 * h * np.array(k2_vec))
            k4_vec = f(t0 + h, y + h * np.array(k3_vec))
            y += (h / 6.0) * (np.array(k1_vec) + 2 * np.array(k2_vec) +
                               2 * np.array(k3_vec) + np.array(k4_vec))
            t0 += h

        # Build a mock result object to stay compatible with the original interface
        class Result:
            def __init__(self, y):
                self.y = y
                self.success = True
                self.message = ""

        return Result(y.reshape(3, 1))