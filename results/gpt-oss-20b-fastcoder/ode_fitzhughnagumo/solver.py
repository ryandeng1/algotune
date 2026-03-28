from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
import numba

class Solver:
    @numba.njit(fastmath=True)
    def _fitzhugh_nagumo(self, t, y, a, b, c, I):
        v, w = y[0], y[1]
        dv_dt = v - v**3 / 3.0 - w + I
        dw_dt = a * (b * v - c * w)
        return np.array([dv_dt, dw_dt], dtype=np.float64)

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True) -> Any:
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        # Pull constants once for speed
        a, b, c, I = params["a"], params["b"], params["c"], params["I"]

        def ode_func(t, y):
            return self._fitzhugh_nagumo(t, y, a, b, c, I)

        t_eval = np.linspace(t0, t1, 1000) if debug else None

        # Using a stable solver with moderate tolerances
        return solve_ivp(
            ode_func,
            (t0, t1),
            y0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-8,
            dense_output=debug,
        )