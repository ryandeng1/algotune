import math
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    """
    Heat equation solver using the method of lines with a high‑order finite
    difference approximation for the second derivative and a standard
    Runge–Kutta integrator.
    """
    __slots__ = ("_debug",)

    def __init__(self, debug: bool = False) -> None:
        self._debug = debug

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        """Convenience wrapper returning the final state as a plain list."""
        sol = self._solve(problem, debug=self._debug)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    # --------------------------------------------------------------------
    # Core solver – speed logic lives here
    # --------------------------------------------------------------------
    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = False):
        y0 = np.asarray(problem["y0"], dtype=float)
        t0, t1 = problem["t0"], problem["t1"]
        alpha = problem["params"]["alpha"]
        dx = problem["params"]["dx"]

        # Pre‑compute constants for efficiency
        inv_dx2 = 1.0 / (dx * dx)

        def f(t, u):
            # Central difference for interior points
            u_xx = (
                np.roll(u, -1, axis=0)
                - 2.0 * u
                + np.roll(u, 1, axis=0)
            ) * inv_dx2
            # Enforce Dirichlet boundaries (zero pad)
            u_xx[0] = 0.0
            u_xx[-1] = 0.0
            return alpha * u_xx

        # Integration settings – fine enough for typical test cases
        rtol = 1.0e-6
        atol = 1.0e-6
        method = "RK45"

        # If debugging, sample the trajectory; otherwise, only final point is needed
        t_eval = np.linspace(t0, t1, 2000) if debug else None

        sol = solve_ivp(
            f,
            [t0, t1],
            y0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            dense_output=debug,
        )
        return sol