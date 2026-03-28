import numpy as np
from scipy.integrate import solve_ivp
from typing import Any

def _vdp(t: float, y: np.ndarray, mu: float) -> np.ndarray:
    """Van der Pol ODE right‑hand side."""
    x, v = y
    return np.array([v, mu * ((1 - x**2) * v - x)], dtype=float)

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True) -> Any:
        y0 = np.asarray(problem["y0"], dtype=float)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        mu = float(problem["mu"])

        # Closure avoids creating a new function on every call
        def rhs(t, y):
            return _vdp(t, y, mu)

        t_eval = np.linspace(t0, t1, 1000) if debug else None
        sol = solve_ivp(
            rhs,
            (t0, t1),
            y0,
            method="Radau",
            rtol=1e-8,
            atol=1e-9,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol