from typing import Any, Dict, List
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    """
    Optimised solver for the Lotka‑Volterra system.
    """

    def __init__(self) -> None:
        # Nothing to initialise once per instance, kept for conformity.
        pass

    def _setup_odefun(self, params: Dict[str, float]):
        """Return a fast, outer‑closure ODE function using only local variable lookup."""
        alpha = params["alpha"]
        beta = params["beta"]
        delta = params["delta"]
        gamma = params["gamma"]

        def odefun(t: float, y: np.ndarray) -> np.ndarray:
            x, y_val = y[0], y[1]
            dx = alpha * x - beta * x * y_val
            dy = delta * x * y_val - gamma * y_val
            return np.array([dx, dy], dtype=y.dtype)

        return odefun

    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, List[float]]:
        """
        Compute the solution of the Lotka‑Volterra equations at t1.
        The function returns the final state vector as a list of floats.
        """
        # Extract data once in a fast manner
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        # Create a local ODE function once per call
        odefun = self._setup_odefun(params)

        # Configure solver – the default tolerances of SciPy are already quite tight
        sol = solve_ivp(
            fun=odefun,
            t_span=(t0, t1),
            y0=y0,
            method="RK45",
            rtol=1e-10,
            atol=1e-10,
            vectorized=False,  # vectorization not needed for 2‑D system
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # Return the last state as a list of floats
        return sol.y[:, -1].tolist()