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

        def heat_equation(t, u):
            alpha = params['alpha']
            dx = params['dx']
            u_padded = np.pad(u, 1, mode='constant', constant_values=0)
            u_xx = (u_padded[2:] - 2 * u_padded[1:-1] + u_padded[:-2]) / dx ** 2
            du_dt = alpha * u_xx
            return du_dt
        rtol = 1e-06
        atol = 1e-06
        method = 'RK45'
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        sol = solve_ivp(heat_equation, [t0, t1], y0, method=method, rtol=rtol, atol=atol, t_eval=t_eval, dense_output=debug)
        return sol
