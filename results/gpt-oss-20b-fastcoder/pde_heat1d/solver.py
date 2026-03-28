from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True) -> Any:
        y0 = np.array(problem['y0'], dtype=np.float64)
        t0, t1 = problem['t0'], problem['t1']
        params = problem['params']
        alpha, dx = params['alpha'], params['dx']

        # Heat equation derivative using centred difference with homogeneous Dirichlet BC
        def heat_equation(_: float, u: np.ndarray) -> np.ndarray:
            # u[0] and u[-1] are boundary nodes set to zero implicitly
            u_left = np.concatenate([np.zeros(1), u[:-1]])
            u_right = np.concatenate([u[1:], np.zeros(1)])
            lap = (u_right - 2 * u + u_left) / dx ** 2
            return alpha * lap

        # Integrate with SciPy only once; no dense output needed for speed
        sol = solve_ivp(
            heat_equation,
            [t0, t1],
            y0,
            method='RK45',
            rtol=1e-6,
            atol=1e-6,
            t_eval=None if not debug else np.linspace(t0, t1, 1000),
            dense_output=False,
        )
        return sol