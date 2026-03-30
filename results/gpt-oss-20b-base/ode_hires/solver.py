import numpy as np
from scipy.integrate import solve_ivp
import numba

# JIT‑compiled right‑hand side for maximal performance
@numba.njit
def _hires(t, y, constants):
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = constants
    y1, y2, y3, y4, y5, y6, y7, y8 = y
    f1 = -c1 * y1 + c2 * y2 + c3 * y3 + c4
    f2 = c1 * y1 - c5 * y2
    f3 = -c6 * y3 + c2 * y4 + c7 * y5
    f4 = c3 * y2 + c1 * y3 - c8 * y4
    f5 = -c9 *  y5 + c2 * y6 + c2 * y7
    f6 = -c10 * y6 * y8 + c11 * y4 + c1 * y5 - c2 * y6 + c11 * y7
    f7 = c10 * y6 * y8 - c12 * y7
    f8 = -c10 * y6 * y8 + c12 * y7
    return np.array([f1, f2, f3, f4, f5, f6, f7, f8])

class Solver:
    """
    Solver for a stiff ODE system using scipy's Radau method.
    The right‑hand side is compiled with numba for speed.
    """

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        """
        Solves the ODE defined in *problem* and returns the final state as a list.
        Raises RuntimeError if the integration fails.
        """
        sol = self._solve(problem, debug=False)
        if sol.success:
            # Convert the last time step to Python list for the expected output type
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True):
        """
        Internal routine that builds the integrator and calls scipy's `solve_ivp`.
        """
        y0 = np.asarray(problem["y0"], dtype=float)
        t0, t1 = problem["t0"], problem["t1"]
        constants = np.asarray(problem["constants"], dtype=float)

        # Wrapper that forwards the constants to the JIT function
        def righthand(t, y):
            return _hires(t, y, constants)

        # No intermediate evaluations when not debugging
        t_eval = np.linspace(t0, t1, 1000) if debug else None

        return solve_ivp(
            righthand,
            (t0, t1),
            y0,
            method="Radau",
            rtol=1e-10,
            atol=1e-9,
            t_eval=t_eval,
            dense_output=debug,
        )