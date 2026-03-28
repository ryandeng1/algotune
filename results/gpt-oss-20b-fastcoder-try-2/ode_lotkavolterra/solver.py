from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

class Solver:
    @staticmethod
    @njit
    def _lotka_numba(t, y, alpha, beta, delta, gamma):
        x, yv = y[0], y[1]
        return np.array([alpha * x - beta * x * yv,
                         delta * x * yv - gamma * yv])

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True) -> Any:
        y0 = np.array(problem['y0'], dtype=np.float64)
        t0, t1 = problem['t0'], problem['t1']
        params = problem['params']
        alpha, beta, delta, gamma = (
            params['alpha'], params['beta'], params['delta'], params['gamma']
        )

        def lotka(t, y):
            return self._lotka_numba(t, y, alpha, beta, delta, gamma)

        rtol, atol = 1e-10, 1e-10
        method = 'RK45'
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        return solve_ivp(
            lotka, [t0, t1], y0,
            method=method, rtol=rtol, atol=atol,
            t_eval=t_eval, dense_output=debug
        )