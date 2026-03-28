from typing import Any
import numpy as np

class Solver:
    """
    A fast, deterministic solver for the 1‑D heat equation with homogeneous Dirichlet
    boundary conditions.  The implementation uses a fully vectorised explicit
    Euler scheme and does *not* call `scipy.integrate.solve_ivp`, which is a
    significant source of overhead for stiff problems.  The time step is chosen
    automatically to respect the Courant‑Friedrichs‑Lewy (CFL) condition for
    stability:
            dt <= dx**2 / (2 * alpha)
    The solver returns the state at time `t1`, matching the interface of
    `solve_ivp` when `dense_output=True` (i.e. the last time point in the
    integration).
    """

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        """Return the state of the PDE at the final time."""
        sol = self._solve(problem, debug=False)
        return sol

    def _solve(self, problem: dict[str, np.ndarray | float], debug=True) -> Any:
        # Extract problem parameters
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        alpha = float(params["alpha"])
        dx = float(params["dx"])

        # Choose time step according to the CFL condition.
        dt = dx**2 / (2 * alpha)  # stability limit for explicit Euler
        n_steps = int(np.ceil((t1 - t0) / dt))
        dt = (t1 - t0) / n_steps  # adjust to reach t1 exactly

        # Pad the field once, then update the interior each step.
        u = np.empty_like(y0)
        for step in range(n_steps):
            # Forward Euler update for interior points
            u[1:-1] = (
                y0[1:-1]
                - alpha * dt / dx**2
                * (y0[2:] - 2 * y0[1:-1] + y0[:-2])
                + 0.0  # boundary terms are zero due to Dirichlet conditions
            )
            # Update boundaries (homogeneous Dirichlet: zero)
            u[0] = 0.0
            u[-1] = 0.0
            y0[:] = u  # prepare for next step

        # Return the final state as a list just like the original interface
        return {"y": y0.tolist()}