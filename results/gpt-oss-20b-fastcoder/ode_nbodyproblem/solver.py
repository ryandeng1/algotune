from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

# ----------------------------------------------------------------------
# Numba‑accelerated core routine
# ----------------------------------------------------------------------
@njit
def _accels_timestep(positions, masses, softening):
    """
    Compute 3‑D N‑body gravitational accelerations for a full time step.
    """
    n = positions.shape[0]
    acc = np.empty_like(positions)
    for i in range(n):
        a_i = np.zeros(3, dtype=np.float64)
        pos_i = positions[i]
        for j in range(n):
            if i == j:          # skip self interaction
                continue
            r = positions[j] - pos_i
            dist_sq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + softening * softening
            inv_r3 = masses[j] / (dist_sq * np.sqrt(dist_sq))
            a_i[0] += inv_r3 * r[0]
            a_i[1] += inv_r3 * r[1]
            a_i[2] += inv_r3 * r[2]
        acc[i] = a_i
    return acc

# ----------------------------------------------------------------------
# Main solver class
# ----------------------------------------------------------------------
class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: dict[str, np.ndarray | float], debug=True) -> Any:
        y0 = np.asarray(problem['y0'], dtype=np.float64)
        t0, t1 = problem['t0'], problem['t1']
        masses = np.asarray(problem['masses'], dtype=np.float64)
        softening = float(problem['softening'])
        num_bodies = int(problem['num_bodies'])
        nvars = num_bodies * 6          # positions + velocities

        # ------------------------------------------------------------------
        # ODE system
        # ------------------------------------------------------------------
        def nbody_problem(t, y):
            # view y as 2D (num_bodies, 6) then split
            y_reshaped = y.reshape(num_bodies, 6)
            positions = y_reshaped[:, :3]
            velocities = y_reshaped[:, 3:]

            # derivatives
            dp_dt = velocities.reshape(-1)                     # dpos/dt = vel
            accelerations = _accels_timestep(positions, masses, softening)
            dv_dt = accelerations.reshape(-1)                 # dvel/dt = accel
            return np.concatenate([dp_dt, dv_dt])

        # ------------------------------------------------------------------
        # Integration options
        # ------------------------------------------------------------------
        rtol, atol = 1e-8, 1e-8
        method = 'RK45'
        t_eval = None
        if debug:
            # produce 1000 equally spaced evaluation points for debugging
            t_eval = np.linspace(t0, t1, 1000)

        return solve_ivp(nbody_problem,
                         [t0, t1],
                         y0,
                         method=method,
                         rtol=rtol,
                         atol=atol,
                         t_eval=t_eval,
                         dense_output=debug)