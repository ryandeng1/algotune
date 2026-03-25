# solver.py
import numpy as np

class Solver:
    """
    A lightweight high‑performance N‑body solver using a fixed‑step
    Velocity‑Verlet (Leapfrog) integrator.  It is deliberately simple,
    avoiding external dependencies beyond NumPy.
    """

    @staticmethod
    def _pairwise_accelerations(positions: np.ndarray, masses: np.ndarray, eps: float) -> np.ndarray:
        """
        Vectorised computation of pairwise gravitational accelerations.
        Parameters
        ----------
        positions : (N, 3) float
            Current positions of all bodies.
        masses : (N,) float
            Mass of each body.
        eps : float
            Softening length.
        Returns
        -------
        accelerations : (N, 3) float
            Acceleration vector for each body.
        """
        # differences r_j - r_i for all pairs (N, N, 3)
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        # squared distances with softening
        dist_sq = np.sum(diff**2, axis=2) + eps**2
        # inverse cubed distance
        inv_dist3 = 1.0 / (dist_sq * np.sqrt(dist_sq))
        # multiply by masses of the attracting bodies
        mass_factors = masses[np.newaxis, :] * inv_dist3
        # sum over j ≠ i
        accel = np.sum(mass_factors[:, :, np.newaxis] * diff, axis=1)
        return accel

    @classmethod
    def solve(cls, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        """
        Compute the final state of the N‑body system at t1.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
                - t0: float
                - t1: float
                - y0: list[float]  (initial positions and velocities)
                - masses: list[float]
                - softening: float
                - num_bodies: int

        Returns
        -------
        dict[str, list[float]]
            The final state vector as a flat list (same layout as y0).
        """
        # Parse problem
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        y0 = np.array(problem["y0"], dtype=np.float64)
        masses = np.array(problem["masses"], dtype=np.float64)
        eps = float(problem["softening"])
        N = int(problem["num_bodies"])

        # Reshape y0 into positions and velocities
        pos = y0[:N * 3].reshape(N, 3)
        vel = y0[N * 3:].reshape(N, 3)

        # Integration parameters
        dt = 1e-3  # fixed step size: 1000 steps per unit time
        steps = int(np.ceil((t1 - t0) / dt))
        dt = (t1 - t0) / steps  # exact step to reach t1

        # Leapfrog (Velocity‑Verlet) integration
        # First half step for velocities
        accel = cls._pairwise_accelerations(pos, masses, eps)
        vel += 0.5 * dt * accel

        for _ in range(steps):
            # Full position update
            pos += dt * vel
            # Full acceleration update
            accel = cls._pairwise_accelerations(pos, masses, eps)
            # Full velocity update
            vel += dt * accel

        # Correct the last half‑step for velocities
        vel -= 0.5 * dt * accel

        # Flatten result
        y_final = np.concatenate([pos.ravel(), vel.ravel()]).tolist()
        return y_final
