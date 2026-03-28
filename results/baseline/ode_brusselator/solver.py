from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1]
        else:
            raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: dict[str, np.ndarray | float], debug=True) -> Any:
        y0 = np.array(problem['y0'])
        t0, t1 = (problem['t0'], problem['t1'])
        params = problem['params']

        def brusselator(t, y):
            X, Y = y
            A = params['A']
            B = params['B']
            dX_dt = A + X ** 2 * Y - (B + 1) * X
            dY_dt = B * X - X ** 2 * Y
            return np.array([dX_dt, dY_dt])
        rtol = 1e-08
        atol = 1e-08
        method = 'RK45'
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        sol = solve_ivp(brusselator, [t0, t1], y0, method=method, rtol=rtol, atol=atol, t_eval=t_eval, dense_output=debug)
        return sol
