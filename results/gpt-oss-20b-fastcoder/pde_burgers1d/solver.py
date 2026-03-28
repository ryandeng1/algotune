import numpy as np
from numba import njit

class Solver:
    # Solve ODE using a single-step RK4 (explicit) for speed.
    # The time step is fixed to 1000 steps for a reasonable accuracy.
    @staticmethod
    @njit
    def _step(u, dx, nu):
        n = u.shape[0]
        u_pad = np.empty(n + 2, dtype=u.dtype)
        u_pad[0] = 0.0
        u_pad[1:-1] = u
        u_pad[-1] = 0.0

        diff = (u_pad[2:] - 2 * u_pad[1:-1] + u_pad[:-2]) / (dx ** 2)
        forward = (u_pad[2:] - u_pad[1:-1]) / dx
        backward = (u_pad[1:-1] - u_pad[:-2]) / dx

        adv = np.where(u_pad[1:-1] >= 0,
                       u_pad[1:-1] * backward,
                       u_pad[1:-1] * forward)
        return -adv + nu * diff

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        y0 = np.asarray(problem['y0'], dtype=np.float64)
        t0, t1 = problem['t0'], problem['t1']
        params = problem['params']
        nu = params['nu']
        dx = params['dx']

        nt = 1000            # fixed number of steps for speed
        dt = (t1 - t0) / nt

        y = y0.copy()
        for _ in range(nt):
            k1 = self._step(y, dx, nu)
            k2 = self._step(y + 0.5 * dt * k1, dx, nu)
            k3 = self._step(y + 0.5 * dt * k2, dx, nu)
            k4 = self._step(y + dt * k3, dx, nu)
            y += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        return y.tolist()