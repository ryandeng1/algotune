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
        y0 = np.asarray(problem['y0'], dtype=float).copy()
        t0, t1 = problem['t0'], problem['t1']
        masses = np.asarray(problem['masses'], dtype=float)
        softening = float(problem['softening'])
        num_bodies = int(problem['num_bodies'])
        dim = 3  # 3‑D space

        # Pre‑computations that stay constant during integration
        inv_masses = 1.0 / masses
        shape = (num_bodies, dim)

        def nbodyproblem(t, y):
            # Split positions and velocities
            pos = y[:num_bodies * dim].reshape(shape)
            vel = y[num_bodies * dim :].reshape(shape)

            # Velocity derivative is simply the current velocity
            dpdt = vel.reshape(-1)

            # Compute pairwise vectors r_ij = r_j - r_i
            r = pos[None, :, :] - pos[:, None, :]  # shape (n, n, 3)

            # Distance squared + softening term
            dist2 = np.sum(r ** 2, axis=2) + softening ** 2  # shape (n, n)

            # Avoid self‑interaction by zeroing diagonal
            np.fill_diagonal(dist2, np.inf)

            # Compute 1 / r^3 term
            inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))

            # Scale by masses of the attracting body
            acc = (inv_dist3 * masses[None, :]) @ r  # shape (n, 3)

            # Velocity derivative is the acceleration
            dvdt = acc.reshape(-1)

            return np.concatenate([dpdt, dvdt])

        rtol, atol = 1e-8, 1e-8
        method = 'RK45'
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        sol = solve_ivp(
            nbodyproblem,
            [t0, t1],
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol