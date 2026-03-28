from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

class Solver:
    # compile the ODE routine once
    @staticmethod
    @njit
    def _vdp(t, y, mu):
        x, v = y[0], y[1]
        dx_dt = v
        dv_dt = mu * ((1.0 - x * x) * v - x)
        return np.array([dx_dt, dv_dt])

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: dict[str, np.ndarray | float], debug=True) -> Any:
        y0 = np.asarray(problem['y0'], dtype=np.float64)
        t0, t1 = float(problem['t0']), float(problem['t1'])
        mu = float(problem['mu'])
        # wrapper to embed mu in the numba function
        def ode_func(t, y):
            return self._vdp(t, y, mu)
        rtol = 1e-8
        atol = 1e-9
        method = 'Radau'
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        sol = solve_ivp(ode_func, [t0, t1], y0, method=method, rtol=rtol, atol=atol,
                        t_eval=t_eval, dense_output=debug)
        return sol