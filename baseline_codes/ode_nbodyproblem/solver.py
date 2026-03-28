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
        masses = np.array(problem['masses'])
        softening = problem['softening']
        num_bodies = problem['num_bodies']

        def nbodyproblem(t, y):
            positions = y[:num_bodies * 3].reshape(num_bodies, 3)
            velocities = y[num_bodies * 3:].reshape(num_bodies, 3)
            dp_dt = velocities.reshape(-1)
            accelerations = np.zeros_like(positions)
            for i in range(num_bodies):
                for j in range(num_bodies):
                    if i != j:
                        r_ij = positions[j] - positions[i]
                        dist_squared = np.sum(r_ij ** 2) + softening ** 2
                        factor = masses[j] / (dist_squared * np.sqrt(dist_squared))
                        accelerations[i] += factor * r_ij
            dv_dt = accelerations.reshape(-1)
            derivatives = np.concatenate([dp_dt, dv_dt])
            return derivatives
        rtol = 1e-08
        atol = 1e-08
        method = 'RK45'
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        sol = solve_ivp(nbodyproblem, [t0, t1], y0, method=method, rtol=rtol, atol=atol, t_eval=t_eval, dense_output=debug)
        return sol
