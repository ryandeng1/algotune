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
        F = float(problem['F'])

        def lorenz96(t, x):
            N = len(x)
            dxdt = np.zeros_like(x)
            ip1 = np.roll(np.arange(N), -1)
            im1 = np.roll(np.arange(N), 1)
            im2 = np.roll(np.arange(N), 2)
            dxdt = (x[ip1] - x[im2]) * x[im1] - x + F
            return dxdt
        rtol = 1e-08
        atol = 1e-08
        method = 'RK45'
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        sol = solve_ivp(lorenz96, [t0, t1], y0, method=method, rtol=rtol, atol=atol, t_eval=t_eval, dense_output=debug)
        return sol
