from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    # -------------------------------------------------------------
    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True) -> Any:
        # Extract data once for speed
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        alpha, dx = params["alpha"], params["dx"]

        # Pre‑allocate padded array structure to avoid reallocating in each step
        m = y0.size
        padded = np.empty(m + 2, dtype=np.float64)

        def heat_eq(t: float, u: np.ndarray) -> np.ndarray:
            # Apply fixed Dirichlet BCs (u[0] = u[-1] = 0) via padding
            padded[0], padded[1:-1], padded[-1] = 0.0, u, 0.0
            u_xx = (padded[2:] - 2 * padded[1:-1] + padded[:-2]) / (dx * dx)
            return alpha * u_xx

        rtol, atol = 1e-6, 1e-6
        method = "RK45"
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        return solve_ivp(
            heat_eq,
            (t0, t1),
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )