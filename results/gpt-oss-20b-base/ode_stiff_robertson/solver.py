from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

class Solver:
    @staticmethod
    @njit(nopython=True)
    def _rober_ro(t, y, k1, k2, k3):
        y1, y2, y3 = y[0], y[1], y[2]
        f0 = -k1 * y1 + k3 * y2 * y3
        f1 = k1 * y1 - k2 * y2 * y2 - k3 * y2 * y3
        f2 = k2 * y2 * y2
        return np.array([f0, f1, f2])

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: dict[str, np.ndarray | float], debug=True) -> Any:
        y0 = np.array(problem['y0'], dtype=np.float64)
        t0, t1 = float(problem['t0']), float(problem['t1'])
        k = tuple(problem['k'])
        k1, k2, k3 = k[0], k[1], k[2]

        def rober_scalar(t, y):
            return self._rober_ro(t, y, k1, k2, k3)

        rtol = 1e-11
        atol = 1e-9
        method = 'Radau'
        if debug:
            t_eval = np.clip(
                np.exp(np.linspace(np.log(1e-6), np.log(t1), 1000)),
                t0, t1
            )
        else:
            t_eval = None

        sol = solve_ivp(
            rober_scalar,
            [t0, t1],
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug
        )
        return sol