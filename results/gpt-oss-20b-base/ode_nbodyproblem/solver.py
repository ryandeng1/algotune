# solver.py
import numpy as np
from scipy.integrate import solve_ivp
import numba

class Solver:
    """
    A fast N‑body solver based on scipy's solve_ivp and a Numba‑accelerated
    force calculation.  The solver is deterministic and returns the state at
    the final time step.
    """

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        """
        Solve the N‑body problem and return the state vector at `t1`.

        Parameters
        ----------
        problem : dict
            Dictionary containing the following keys:
            - 'y0'      : Initial state vector (positions followed by velocities).
            - 't0'      : Initial time.
            - 't1'      : Final time.
            - 'masses'  : Array of masses, shape (N,).
            - 'softening': Softening length.
            - 'num_bodies': Integer, number of bodies.
        Returns
        -------
        dict
            Dictionary with a single key `'y'` containing a list of the final
            state values.
        """
        sol = self._solve(problem, debug=False)
        if sol.success:
            return {"y": sol.y[:, -1].tolist()}
        raise RuntimeError(f"Solver failed: {sol.message}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    @numba.njit
    def _forces(num_bodies, positions, masses, softening):
        """
        Compute accelerations for all bodies using Newton's law
        with optional softening.
        """
        accelerations = np.zeros((num_bodies, 3), dtype=np.float64)
        for i in range(num_bodies):
            p_i = positions[i]
            for j in range(i + 1, num_bodies):
                p_j = positions[j]
                r_ij = p_j - p_i
                dist_sq = np.dot(r_ij, r_ij) + softening * softening
                inv_r3 = 1.0 / (dist_sq * np.sqrt(dist_sq))
                factor = masses[j] * inv_r3
                accelerations[i] += factor * r_ij
                # Newton's third law
                accelerations[j] -= masses[i] * inv_r3 * r_ij
        return accelerations

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True) -> Any:
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        masses = np.asarray(problem["masses"], dtype=np.float64)
        softening = problem["softening"]
        num_bodies = int(problem["num_bodies"])

        # The following closure will be called by solve_ivp.
        def nbody(t, y):
            # Positions & velocities are interleaved
            pos = y[: num_bodies * 3].reshape(num_bodies, 3)
            vel = y[num_bodies * 3 :].reshape(num_bodies, 3)

            # Acceleration calculation is numba‑optimized
            acc = self._forces(num_bodies, pos, masses, softening)

            # Derivative vector: dp/dt = v, dv/dt = a
            return np.concatenate([vel.ravel(), acc.ravel()])

        t_eval = np.linspace(t0, t1, 1000) if debug else None
        sol = solve_ivp(
            nbody,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol