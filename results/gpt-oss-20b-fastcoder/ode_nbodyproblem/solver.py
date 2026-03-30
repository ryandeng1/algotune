# solver.py
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

@njit
def _nbody_deriv(t, y, num_bodies, masses, softening):
    """Numba‑compiled derivative function for the N‑body problem."""
    # Split positions and velocities
    total = num_bodies * 3
    positions = y[:total].reshape(num_bodies, 3)
    velocities = y[total:].reshape(num_bodies, 3)

    dp_dt = velocities.ravel()          # du/dt = v
    accelerations = np.zeros_like(positions)

    for i in range(num_bodies):
        pi = positions[i]
        for j in range(num_bodies):
            if i != j:
                # Relative position
                r0 = positions[j, 0] - pi[0]
                r1 = positions[j, 1] - pi[1]
                r2 = positions[j, 2] - pi[2]
                dist2 = r0 * r0 + r1 * r1 + r2 * r2 + softening * softening
                inv = 1.0 / (dist2 * np.sqrt(dist2))
                factor = masses[j] * inv
                accelerations[i, 0] += factor * r0
                accelerations[i, 1] += factor * r1
                accelerations[i, 2] += factor * r2

    dv_dt = accelerations.ravel()       # d(v)/dt = a
    return np.concatenate((dp_dt, dv_dt))


class Solver:
    """
    Optimised N‑body solver.

    The derivative function is JIT‑compiled with numba and free of Python
    loops, which dramatically speeds up long integrations.  The public
    interface remains unchanged.
    """

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True):
        # Extract and cast problem parameters to float64 numpy arrays
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        masses = np.array(problem["masses"], dtype=np.float64)
        softening = float(problem["softening"])
        num_bodies = int(problem["num_bodies"])

        # Set integration parameters
        rtol, atol = 1e-8, 1e-8
        method = "RK45"
        t_eval = (
            np.linspace(t0, t1, 1000)
            if debug
            else None
        )

        # Wrap the numba‑compiled derivative into a lambda that passes extra
        # arguments to `_nbody_deriv`.
        def deriv(t, y):
            return _nbody_deriv(t, y, num_bodies, masses, softening)

        # Run the ODE solver
        sol = solve_ivp(
            deriv,
            [t0, t1],
            y0,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            method=method,
            dense_output=debug,
        )
        return sol