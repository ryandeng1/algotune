from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem)
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = False) -> Any:
        y0 = np.array(problem['y0'])
        t0, t1 = problem['t0'], problem['t1']
        params = problem['params']
        nu = params['nu']
        dx = params['dx']
        nx = y0.size

        # Pre‑compute constants
        invdx = 1.0 / dx
        invdx2 = invdx * invdx
        nu_d = nu * invdx2

        def burgers_equation(t, u):
            # Pad with zeros at both ends via np.concatenate for speed
            u_pad = np.empty(nx + 2, dtype=u.dtype)
            u_pad[0] = 0.0
            u_pad[1:-1] = u
            u_pad[-1] = 0.0

            u_center = u_pad[1:-1]
            diff_forward = u_pad[2:] - u_pad[1:-1]
            diff_backward = u_pad[1:-1] - u_pad[:-2]

            # Advection term with upwind
            advection = np.where(u_center >= 0,
                                 u_center * diff_backward,
                                 u_center * diff_forward)
            advection *= invdx

            diffusion = (u_pad[2:] - 2 * u_pad[1:-1] + u_pad[:-2]) * invdx2

            return -advection + nu_d * diffusion

        # Integration without intermediate evaluations
        sol = solve_ivp(
            burgers_equation,
            (t0, t1),
            y0,
            method='RK45',
            rtol=1e-6,
            atol=1e-6,
            t_eval=None,          # only final state needed
            dense_output=False
        )
        return sol