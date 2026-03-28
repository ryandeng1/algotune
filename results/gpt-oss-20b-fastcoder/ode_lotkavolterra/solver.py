from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True):
        y0 = np.asarray(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        # Pull parameters to local scope for speed
        alpha, beta, delta, gamma = (
            params["alpha"],
            params["beta"],
            params["delta"],
            params["gamma"],
        )

        def lotka_volterra(t: float, y: np.ndarray):
            x, y_ = y
            dx_dt = alpha * x - beta * x * y_
            dy_dt = delta * x * y_ - gamma * y_
            return np.array([dx_dt, dy_dt], dtype=y.dtype)

        t_eval = np.linspace(t0, t1, 1000, dtype=y0.dtype, endpoint=True) if debug else None

        # Use a faster explicit Runge–Kutta method
        return solve_ivp(
            lotka_volterra,
            (t0, t1),
            y0,
            method="DOP853",
            rtol=1e-8,
            atol=1e-8,
            t_eval=t_eval,
            dense_output=debug,
        )