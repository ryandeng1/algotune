# solver.py
from typing import Any, List
import numpy as np
from scipy.integrate import solve_ivp


class Solver:
    """Solver for the HIRES stiff ODE system."""

    @staticmethod
    def _hires(t: float, y: np.ndarray, constants: List[float]) -> np.ndarray:
        """Right‑hand side of the HIRES system."""
        c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = constants
        y1, y2, y3, y4, y5, y6, y7, y8 = y

        f1 = -c1 * y1 + c2 * y2 + c3 * y3 + c4
        f2 = c1 * y1 - c5 * y2
        f3 = -c6 * y3 + c2 * y4 + c7 * y5
        f4 = c3 * y2 + c1 * y3 - c8 * y4
        f5 = -c9 * y5 + c2 * y6 + c2 * y7
        f6 = -c10 * y6 * y8 + c11 * y4 + c1 * y5 - c2 * y6 + c11 * y7
        f7 = c10 * y6 * y8 - c12 * y7
        f8 = -c10 * y6 * y8 + c12 * y7
        return np.array([f1, f2, f3, f4, f5, f6, f7, f8], dtype=np.float64)

    def solve(self, problem: dict[str, Any]) -> List[float]:
        """
        Solve the HIRES ODE system and return the state at t1.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - t0 (float): start time
                - t1 (float): end time
                - y0 (list[float]): initial state (length 8)
                - constants (list[float]): 12 rate constants

        Returns
        -------
        list[float]
            State vector at time t1 (length 8)
        """
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        constants = problem["constants"]
        t0, t1 = problem["t0"], problem["t1"]

        # Use a stiff solver with tight tolerances
        sol = solve_ivp(
            fun=lambda t, y: self._hires(t, y, constants),
            t_span=(t0, t1),
            y0=y0,
            method="BDF",
            rtol=1e-10,
            atol=1e-9,
            vectorized=False,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        return sol.y[:, -1].tolist()
