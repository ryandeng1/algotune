# solver.py
from __future__ import annotations
from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit, int64, float64, types
from numba.typed import List

# -------------------------------------------------------------
# Numerical driver for the Lorenz 96 system
# -------------------------------------------------------------
@njit
def _lorenz96_numba(x: np.ndarray, t: float, N: int, F: float,
                    ip1: np.ndarray, im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    JIT‑compiled Lorenz 96 RHS.  All arguments are simple numerical types
    so the call from SciPy has no Python overhead.
    """
    out = np.empty(N, dtype=float64)
    for i in range(N):
        out[i] = (x[ip1[i]] - x[im2[i]]) * x[im1[i]] - x[i] + F
    return out


class Solver:
    """
    Fast solver for the Lorenz 96 system.
    
    The implementation precomputes the index roll patterns once per
    problem and delegates the RHS evaluation to a Numba
    function, eliminating the Python loop inside `solve_ivp`.
    """

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        """
        Solve the ODE and return the final state
        """
        sol = self._solve(problem, debug=False)
        if sol.success:
            return {"y": sol.y[:, -1].tolist()}
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True) -> Any:
        # ---- Prepare data ----------------------------------------------------
        y0: np.ndarray = np.asarray(problem["y0"], dtype=np.float64, order="C")
        N: int = len(y0)

        t0: float = float(problem["t0"])
        t1: float = float(problem["t1"])
        F: float = float(problem["F"])

        # Pre‑compute the fixed index arrays reused by the numba RHS.
        ip1 = np.roll(np.arange(N, dtype=int64), -1)
        im1 = np.roll(np.arange(N, dtype=int64), 1)
        im2 = np.roll(np.arange(N, dtype=int64), 2)

        # The Lorenz‑96 RHS function that will be passed to solve_ivp.
        # It proxies to the JIT compiled routine and injects the fixed indices.
        def lorenz96(t, x):
            return _lorenz96_numba(x, t, N, F, ip1, im1, im2)

        # Solver options
        method = "DOP853"
        rtol = 1e-8
        atol = 1e-8

        # If debugging, generate analytical solution at 1000 points; otherwise use dense_output.
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        dense_output = debug

        return solve_ivp(
            lorenz96,
            [t0, t1],
            y0,
            method=method,
            t_eval=t_eval,
            dense_output=dense_output,
            rtol=rtol,
            atol=atol,
        )