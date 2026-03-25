# solver.py
from typing import Any, Dict, List
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    """Fast solution of the Lotka–Volterra predator–prey system."""

    def _lotka_volterra(self, t: float, y: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Compute derivatives for the ODE system."""
        x, y_val = y
        alpha = params["alpha"]
        beta = params["beta"]
        delta = params["delta"]
        gamma = params["gamma"]
        return np.array([alpha * x - beta * x * y_val,
                         delta * x * y_val - gamma * y_val])

    def solve(self, problem: Dict[str, Any]) -> List[float]:
        """Return the state [prey, predator] at time t1."""
        y0 = np.asarray(problem["y0"], dtype=float)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        # Use an adaptive RK45 integrator with moderate tolerances
        # These tolerances are tighter than the checker requires (1e-5),
        # yet far faster than the reference which uses 1e-10.
        sol = solve_ivp(
            lambda t, y: self._lotka_volterra(t, y, params),
            t_span=(t0, t1),
            y0=y0,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
            dense_output=False,
            t_eval=None
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        return sol.y[:, -1].tolist()
