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

        def burgers_equation(t, u):
            nu = params['nu']
            dx = params['dx']
            u_padded = np.pad(u, 1, mode='constant', constant_values=0)
            diffusion_term = (u_padded[2:] - 2 * u_padded[1:-1] + u_padded[:-2]) / dx ** 2
            u_centered = u_padded[1:-1]
            du_dx_forward = (u_padded[2:] - u_padded[1:-1]) / dx
            du_dx_backward = (u_padded[1:-1] - u_padded[:-2]) / dx
            advection_term = np.where(u_centered >= 0, u_centered * du_dx_backward, u_centered * du_dx_forward)
            du_dt = -advection_term + nu * diffusion_term
            return du_dt
        rtol = 1e-06
        atol = 1e-06
        method = 'RK45'
        t_eval = np.linspace(t0, t1, 100) if debug else None
        sol = solve_ivp(burgers_equation, [t0, t1], y0, method=method, rtol=rtol, atol=atol, t_eval=t_eval, dense_output=debug)
        return sol
