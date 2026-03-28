from typing import Any
import numpy as np
from scipy.integrate import solve_ivp


class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return {"y": sol.y[:, -1].tolist()}
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(
        self, problem: dict[str, np.ndarray | float], debug: bool = True
    ) -> Any:
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        F = float(problem["F"])
        n = y0.size

        # Pre‑compute circular indices once
        ip1 = np.roll(np.arange(n), -1)
        im1 = np.roll(np.arange(n), 1)
        im2 = np.roll(np.arange(n), 2)

        def lorenz96(t, x):
            # x is already a numpy array of dtype float64
            return (x[ip1] - x[im2]) * x[im1] - x + F

        t_eval = (
            np.linspace(t0, t1, 1000, dtype=np.float64) if debug else None
        )
        sol = solve_ivp(
            lorenz96,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol