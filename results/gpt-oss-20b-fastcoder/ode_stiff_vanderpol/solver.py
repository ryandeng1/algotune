from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    """
    Optimised solver for the Van der Pol oscillation problem.  The implementation
    avoids allocation overheads and unnecessary features that are not required
    for the benchmark, resulting in a much faster run‑time.
    """

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        """
        Solve the ODE with the given parameters and return the final state as a list.
        """
        sol = self._solve(problem, debug=False)
        if sol.success:
            # Convert final state from array to list
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True) -> Any:
        """
        Core solver that is called only once per problem instance.
        The `debug` flag switches dense output and evaluation points on/off
        to minimise execution time during scoring.
        """
        # Extract problem data; use local variables to avoid repetitive dict lookups
        y0 = np.asarray(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        mu = float(problem["mu"])

        # Use a compact inline function that returns a 2‑element numpy array.
        # The local variables avoid attribute lookup inside the loop.
        def vdp(t: float, y: np.ndarray) -> np.ndarray:
            x, v = y  # tuple unpacking of 2‑element array
            return np.array([v, mu * ((1 - x * x) * v - x)], dtype=float)

        # Integration parameters chosen for a good trade‑off between speed and
        # accuracy.  The method is simple but robust for the Van der Pol oscillator.
        rtol = 1e-8
        atol = 1e-9
        method = "RK45"

        # Avoid `t_eval` and dense output unless in debug mode.
        t_eval = np.linspace(t0, t1, 1000) if debug else None

        sol = solve_ivp(
            vdp,
            (t0, t1),
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol