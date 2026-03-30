# solver.py
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp


@njit
def _derivative(t: float, y: np.ndarray, constants: np.ndarray) -> np.ndarray:
    """
    Accelerated derivative function.  All scalars are unpacked locally to
    avoid attribute look‑ups in the hot loop.
    """
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = constants

    y1, y2, y3, y4, y5, y6, y7, y8 = y

    f1 = -c1 * y1 + c2 * y2 + c3 * y3 + c4
    f2 = c1 * y1 - c5 * y2
    f3 = -c6 * y3 + c2 * y4 + c7 * y5
    f4 = c3 * y2 + c1 * y3 - c8 * y4
    f5 = -c9 * y5 + c2 * y6 + c2 * y7
    f6 = -c10 * y6 * y8 + c11 * y4 + c1 * y5 - c2 * y6 + c11 * y7
    f7 = c10 * y6 * y8 - c12 * y7
    f8 = -c10 * y6 * y8 + c12 * y7

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8], dtype=np.float64)


class Solver:
    """
    Fast solver for the 8‑dimensional stiff ODE system.
    Uses a JIT‑compiled derivative function and Radau integration.
    """

    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, List[float]]:
        """
        Solve the ODE and return the final state as a plain Python list.
        """
        sol = self._solve(problem, debug=False)
        if sol.success:
            return {"result": sol.y[:, -1].tolist()}
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")

    # ----------------------------------------------------------------------
    def _solve(
        self, problem: Dict[str, np.ndarray | float], debug: bool = True
    ) -> Any:
        """
        Low‑level solver using scipy.integrate.solve_ivp.
        The derivative is JIT compiled with Numba for speed.
        """
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        constants = np.asarray(problem["constants"], dtype=np.float64)

        def func(t, y):
            # Wrapper that passes constants to the JIT compiled function
            return _derivative(t, y, constants)

        kwargs: Dict[str, Any] = {
            "method": "Radau",
            "rtol": 1e-10,
            "atol": 1e-9,
            "dense_output": debug,
        }

        if debug:
            kwargs["t_eval"] = np.linspace(t0, t1, 1000)

        sol = solve_ivp(func, [t0, t1], y0, **kwargs)
        return sol