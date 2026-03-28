from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: dict[str, np.ndarray | float], debug=True) -> Any:
        y0 = np.array(problem['y0'])
        t0, t1 = (float(problem['t0']), float(problem['t1']))
        k = tuple(problem['k'])

        def rober(t, y):
            y1, y2, y3 = y
            k1, k2, k3 = k
            f0 = -k1 * y1 + k3 * y2 * y3
            f1 = k1 * y1 - k2 * y2 ** 2 - k3 * y2 * y3
            f2 = k2 * y2 ** 2
            return np.array([f0, f1, f2])
        rtol = 1e-11
        atol = 1e-09
        method = 'Radau'
        if debug:
            t_eval = np.clip(np.exp(np.linspace(np.log(1e-06), np.log(t1), 1000)), t0, t1)
        else:
            t_eval = None
        sol = solve_ivp(rober, [t0, t1], y0, method=method, rtol=rtol, atol=atol, t_eval=t_eval, dense_output=debug)
        return sol
