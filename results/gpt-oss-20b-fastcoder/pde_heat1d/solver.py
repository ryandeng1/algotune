# solver.py
import numpy as np
from numba import njit

class Solver:
    @staticmethod
    @njit
    def _laplacian(u, dx):
        """
        Compute the second spatial derivative with Dirichlet zero boundary
        conditions using a padded array.
        """
        n = u.shape[0]
        lap = np.empty_like(u)
        # inner points
        lap[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx**2
        # boundaries (Dirichlet=0)
        lap[0] = (u[1] - 2.0 * u[0] + 0) / dx**2
        lap[-1] = (0 - 2.0 * u[-1] + u[-2]) / dx**2
        return lap

    @staticmethod
    @njit
    def _derivative(u, params):
        """Compute du/dt for the heat equation."""
        return params['alpha'] * Solver._laplacian(u, params['dx'])

    def _rk4_step(self, u, h, params):
        """Single step of the classical 4th‑order Runge–Kutta method."""
        k1 = self._derivative(u, params)
        k2 = self._derivative(u + 0.5 * h * k1, params)
        k3 = self._derivative(u + 0.5 * h * k2, params)
        k4 = self._derivative(u + h * k3, params)
        return u + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        """Fast deterministic integrator using a fixed 4th‑order RK scheme."""
        y0 = np.asarray(problem['y0'], dtype=np.float64)
        t0, t1 = problem['t0'], problem['t1']
        params = problem['params']

        # Choose number of steps
        n_steps = 1000  # fixed, matches original t_eval length
        h = (t1 - t0) / n_steps
        u = y0.copy()

        for _ in range(n_steps):
            u = self._rk4_step(u, h, params)

        return u.tolist()