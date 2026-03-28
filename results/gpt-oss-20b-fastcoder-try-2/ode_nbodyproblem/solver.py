import numpy as np
from scipy.integrate import solve_ivp
from numba import njit, prange

class Solver:

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def _nbody_step(t, y, masses, softening, num_bodies):
        # Unpack positions and velocities
        pos = y[:num_bodies * 3].reshape(num_bodies, 3)
        vel = y[num_bodies * 3:].reshape(num_bodies, 3)
        # Core of the equations
        # Compute pairwise separation vectors
        diff = pos[:, None, :] - pos[None, :, :]      # shape (n, n, 3)
        dist_sqr = np.sum(diff ** 2, axis=2) + softening**2   # (n, n)
        inv_dist3 = np.power(dist_sqr, -1.5)         # (n, n)
        np.fill_diagonal(inv_dist3, 0.0)             # zero self-interaction
        acc = np.zeros((num_bodies, 3), dtype=np.float64)
        # Broadcast masses
        m = masses[None, :]
        # Compute accelerations
        for i in prange(num_bodies):
            # Sum over j
            acc[i] = np.sum((m * inv_dist3[:, i, None]) * diff[:, i], axis=0)
        # Derivatives
        dp_dt = vel.reshape(-1)
        dv_dt = acc.reshape(-1)
        return np.concatenate([dp_dt, dv_dt])

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        y0 = problem['y0'].astype(np.float64, copy=False)
        t0, t1 = problem['t0'], problem['t1']
        masses = problem['masses'].astype(np.float64, copy=False)
        softening = problem['softening']
        num_bodies = problem['num_bodies']

        def nbody_problem(t, y):
            return self._nbody_step(t, y, masses, softening, num_bodies)

        # Use a single output step to speed up
        sol = solve_ivp(
            nbody_problem,
            [t0, t1],
            y0,
            method='RK45',
            rtol=1e-8,
            atol=1e-8,
            dense_output=False,
            max_step=(t1 - t0) / 1000  # limit step to keep ODE stable
        )
        if not sol.success:
            raise RuntimeError(f'Solver failed: {sol.message}')
        return sol.y[:, -1].tolist()