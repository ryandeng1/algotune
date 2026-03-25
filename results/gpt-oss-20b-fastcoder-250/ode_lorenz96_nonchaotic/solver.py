# solver.py
import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> List[float]:
        """Solve Lorenz 96 ODE with fixed‐step RK4.

        Parameters
        ----------
        problem : dict
            Keys:
                F   : float  forcing term
                t0  : float  start time
                t1  : float  end time
                y0  : list[float] initial state

        Returns
        -------
        list[float]
            State vector at time t1.
        """
        y = np.asarray(problem["y0"], dtype=np.float64)
        F = float(problem["F"])
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        N = y.size

        # Choose a step that gives good accuracy for non‑chaotic regime.
        dt = 0.01
        n_steps = int(np.ceil((t1 - t0) / dt))
        dt = (t1 - t0) / n_steps  # adjust to hit t1 exactly

        # Pre‑compute index shifts for speed
        ip1 = np.roll(np.arange(N), -1)
        im1 = np.roll(np.arange(N), 1)
        im2 = np.roll(np.arange(N), 2)

        def rhs(x: np.ndarray) -> np.ndarray:
            """Lorenz‑96 RHS."""
            return (x[ip1] - x[im2]) * x[im1] - x + F

        # Fixed‑step RK4
        for _ in range(n_steps):
            k1 = rhs(y)
            k2 = rhs(y + 0.5 * dt * k1)
            k3 = rhs(y + 0.5 * dt * k2)
            k4 = rhs(y + dt * k3)
            y += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return y.tolist()
