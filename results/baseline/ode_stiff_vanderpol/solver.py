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
        mu = float(problem['mu'])

        def vdp(t, y):
            x, v = y
            dx_dt = v
            dv_dt = mu * ((1 - x ** 2) * v - x)
            return np.array([dx_dt, dv_dt])
        rtol = 1e-08
        atol = 1e-09
        method = 'Radau'
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        sol = solve_ivp(vdp, [t0, t1], y0, method=method, rtol=rtol, atol=atol, t_eval=t_eval, dense_output=debug)
        return sol
