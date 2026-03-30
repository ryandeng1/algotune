# solver.py
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

# --------------------------------------------------------------------------- #
# 1.  Numba‑accelerated Lorenz‑96 RHS
# --------------------------------------------------------------------------- #
# The indices used in the ODE will be computed once per problem instance and
# kept as 64‑bit integers for fast Numba lookup.
@njit(nopython=True, cache=True)
def lorenz96_rhs(t, x, ip1, im1, im2, N, F):
    """Return dx/dt for Lorenz‑96 at time t and state x."""
    # x is a 1‑D array of length N
    dxdt = np.empty_like(x)
    # (x[i+1] - x[i-2]) * x[i-1] - x[i] + F
    dxdt = (x[ip1] - x[im2]) * x[im1] - x + F
    return dxdt

# --------------------------------------------------------------------------- #
# 2.  Solver class
# --------------------------------------------------------------------------- #
class Solver:
    """
    Solves the Lorenz‑96 system using scipy.integrate.solve_ivp with a
    Numba‑compiled right‑hand side.  The solver is wrapped in a class so
    the JIT compilation of the kernel happens only once during the first call.
    """

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        """
        Integrate the Lorenz‑96 ODE and return the final state.
        """
        # Delegate to the lower‑level routine
        sol = self._solve(problem, debug=False)
        if sol.success:
            return {"y": sol.y[:, -1].tolist()}
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True):
        """
        Internal routine that prepares indices and calls solve_ivp.
        """
        # --------------------------------------------------------------------- #
        # 1.  Parse problem data
        # --------------------------------------------------------------------- #
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        F = float(problem["F"])

        N = y0.size

        # --------------------------------------------------------------------- #
        # 2.  Precompute integer indices only once
        # --------------------------------------------------------------------- #
        from numpy import arange, roll

        idx = arange(N, dtype=np.int64)
        ip1 = roll(idx, -1)  # i+1
        im1 = roll(idx, 1)   # i-1
        im2 = roll(idx, 2)   # i-2

        # --------------------------------------------------------------------- #
        # 3.  Wrapper so that solve_ivp can call the numba routine
        # --------------------------------------------------------------------- #
        def rhs(t, x):
            return lorenz96_rhs(t, x, ip1, im1, im2, N, F)

        # --------------------------------------------------------------------- #
        # 4.  Solver parameters
        # --------------------------------------------------------------------- #
        method = "RK45"
        rtol = 1e-08
        atol = 1e-08
        t_eval = None if not debug else np.linspace(t0, t1, 1000)

        # --------------------------------------------------------------------- #
        # 5.  Solve the ODE
        # --------------------------------------------------------------------- #
        sol = solve_ivp(
            rhs,
            [t0, t1],
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol