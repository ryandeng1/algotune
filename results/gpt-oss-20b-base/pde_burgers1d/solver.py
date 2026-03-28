import numpy as np
from typing import Any

class Solver:
    """Fast solver for the 1‑D viscous Burgers equation using a
    vectorised 4th‑order Runge–Kutta with fixed time steps.  The
    implementation replaces the scipy integrator to eliminate the
    per‑step Python call overhead and the costly padding operation.
    """

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        # return the final state as a list
        return sol.tolist()

    def _solve(self, problem: dict[str, np.ndarray | float], debug=True) -> Any:
        y0 = np.asarray(problem["y0"], dtype=float)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        nu, dx = params["nu"], params["dx"]

        # Number of fixed time steps (same as the original t_eval grid)
        nsteps = 100 if debug else 100
        dt = (t1 - t0) / nsteps

        # Pre‑compute shift indices for ghost cells (zero padding)
        mx = y0.size
        # Helper to compute diffusive and advective terms with zero boundaries
        def f(u):
            # padding with zeros on both ends
            up = np.empty(mx + 2, dtype=u.dtype)
            up[1:-1] = u
            up[0] = up[-1] = 0.0

            # Laplacian term
            lap = (up[2:] - 2 * up[1:-1] + up[:-2]) / dx ** 2

            # Advection terms (upwind)
            u_c = up[1:-1]
            du_f = (up[2:] - up[1:-1]) / dx
            du_b = (up[1:-1] - up[:-2]) / dx
            adv = np.where(u_c >= 0, u_c * du_b, u_c * du_f)

            return -adv + nu * lap

        # Runge–Kutta 4 with vectorised operations
        u = y0.copy()
        for _ in range(nsteps):
            k1 = f(u)
            k2 = f(u + 0.5 * dt * k1)
            k3 = f(u + 0.5 * dt * k2)
            k4 = f(u + dt * k3)
            u += dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        return u