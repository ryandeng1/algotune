from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    """Highly‑optimised N‑body ODE solver."""

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        """Integrate the N‑body problem and return the final state."""
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug=True) -> Any:
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        masses = np.array(problem["masses"], dtype=np.float64)
        softening = float(problem["softening"])
        num_bodies = int(problem["num_bodies"])

        def nbodyproblem(t, y):
            # Split state vector into positions and velocities
            positions = y[: num_bodies * 3].reshape(num_bodies, 3)
            velocities = y[num_bodies * 3 :].reshape(num_bodies, 3)

            # Derivative of positions = velocities
            dp_dt = velocities.reshape(-1)

            # Vectorised pairwise interaction
            # r i,j = pos_j - pos_i
            r = positions[None, :, :] - positions[:, None, :]  # (N,N,3)
            dist_sq = (r**2).sum(-1) + softening**2                     # (N,N)
            np.fill_diagonal(dist_sq, np.inf)          # ignore self interaction
            inv_dist3 = 1.0 / (dist_sq * np.sqrt(dist_sq))  # (N,N)
            # Multiply by masses of the source bodies
            inv_dist3 *= masses[None, :]              # (N,N)
            # Acceleration of each body: sum over j
            accelerations = (inv_dist3[:, :, None] * r).sum(axis=1)

            dv_dt = accelerations.reshape(-1)

            return np.concatenate([dp_dt, dv_dt])

        # Integration settings
        rtol = 1e-8
        atol = 1e-8
        method = "RK45"
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