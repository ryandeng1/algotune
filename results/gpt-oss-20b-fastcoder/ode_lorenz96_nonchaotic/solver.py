from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True) -> Any:
        y0 = np.array(problem["y0"], dtype=float, copy=False)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        F = float(problem["F"])

        N = y0.size
        # pre‑compute index arrays once
        idx_ip1, idx_im1, idx_im2 = (
            np.arange(N) + 1,  # ip1
            np.arange(N) - 1,  # im1
            np.arange(N) - 2,  # im2
        )
        idx_ip1 %= N
        idx_im1 %= N
        idx_im2 %= N

        def lorenz96(t, x):
            # No allocation beyond the view `x`
            dx = (x[idx_ip1] - x[idx_im2]) * x[idx_im1] - x + F
            return dx

        rtol, atol = 1e-8, 1e-8
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        sol = solve_ivp(
            lorenz96,
            [t0, t1],
            y0,
            method="RK45",
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol