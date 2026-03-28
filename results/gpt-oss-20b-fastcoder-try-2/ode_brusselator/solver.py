from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit, prange

class Solver:
    """
    Optimized solver for the Brusselator ODE system.
    Uses a Numba-accelerated right‑hand side function and
    sensible integration parameters to reduce function call overhead.
    """

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            # Return the last state vector as a plain list of floats
            return {k: list(sol.y[i, -1]) for i, k in enumerate(["X", "Y"])}
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug=True) -> Any:
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        A, B = params["A"], params["B"]

        @njit
        def rhs(t, y):
            # y[0] = X, y[1] = Y
            X, Y = y[0], y[1]
            dX = A + X * X * Y - (B + 1) * X
            dY = B * X - X * X * Y
            return np.array([dX, dY], dtype=np.float64)

        rtol, atol = 1e-8, 1e-8
        max_step = (t1 - t0) / 2000.0  # small fixed step to keep solver fast
        t_eval = np.linspace(t0, t1, 1000) if debug else None

        sol = solve_ivp(
            rhs,
            [t0, t1],
            y0,
            method="RK45",
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol