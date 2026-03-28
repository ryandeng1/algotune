import numpy as np
from typing import Any

class Solver:
    """Fast solver for 1‑D heat equation with homogeneous Dirichlet BCs."""
    
    def __init__(self):
        pass
    
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: dict[str, np.ndarray | float], debug=True) -> Any:
        # ------------------------------------------------------------------
        # Input parsing
        # ------------------------------------------------------------------
        y0 = problem['y0'].astype(np.float64)
        t0, t1 = problem['t0'], problem['t1']
        params = problem['params']
        alpha = params['alpha']
        dx = params['dx']

        # ------------------------------------------------------------------
        # Physics ----------------------------------------------------------------
        # The heat equation discretised in space gives
        #   du/dt = alpha * (u_{i+1} - 2*u_i + u_{i-1}) / dx^2
        # With homogeneous Dirichlet BCs the boundary points stay 0.
        # ------------------------------------------------------------------
        n = y0.size
        # Stability limits for explicit Euler: dt <= dx^2/(2*alpha)
        dt_max = dx * dx / (2 * alpha)
        # Choose number of steps so that all timesteps are equal and the last
        # step lands on t1 (or at least at or beyond t1). Use a safety factor.
        # In debug mode we output intermediate solution.
        num_steps = int(np.ceil((t1 - t0) / dt_max))
        dt = (t1 - t0) / num_steps
        nt = num_steps

        # ------------------------------------------------------------------
        # Pre‑allocate arrays
        # ------------------------------------------------------------------
        u = y0.copy()
        if debug:
            y_out = np.empty((n, nt + 1))
            y_out[:, 0] = u
        # ------------------------------------------------------------------
        # One‑dimensional Laplacian with Dirichlet boundaries
        # ------------------------------------------------------------------
        inv_dx2 = 1.0 / (dx * dx)

        # ------------------------------------------------------------------
        # Time marching – explicit Euler (vectorised)
        # ------------------------------------------------------------------
        for k in range(1, nt + 1):
            # second derivative: pad with zeros for boundaries
            u_pad = np.pad(u, 1, mode='constant')
            u_xx = (u_pad[2:] - 2 * u_pad[1:-1] + u_pad[:-2]) * inv_dx2
            u += alpha * dt * u_xx
            if debug:
                y_out[:, k] = u

        # ------------------------------------------------------------------
        # Build simple result object – mimic scipy's result for API match
        # ------------------------------------------------------------------
        class Result:
            success = True
            t = np.linspace(t0, t1, nt + 1)
            y = y_out if debug else u[:, None]

        return Result()