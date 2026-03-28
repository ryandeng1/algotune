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
        t0, t1 = (problem['t0'], problem['t1'])
        params = problem['params']

        def fitzhugh_nagumo(t, y):
            v, w = y
            a = params['a']
            b = params['b']
            c = params['c']
            I = params['I']
            dv_dt = v - v ** 3 / 3 - w + I
            dw_dt = a * (b * v - c * w)
            return np.array([dv_dt, dw_dt])
        rtol = 1e-08
        atol = 1e-08
        method = 'RK45'
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        sol = solve_ivp(fitzhugh_nagumo, [t0, t1], y0, method=method, rtol=rtol, atol=atol, t_eval=t_eval, dense_output=debug)
        return sol
