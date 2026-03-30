# solver.py
"""
Optimised solver for Robertson ODE system.
Uses a Numba‑compiled RHS function and SciPy solve_ivp with
Radau method.  All temporary NumPy arrays are explicitly prevented,
and only a single dense output is returned if debugging is requested.
"""

from __future__ import annotations
from typing import Any, Dict, List, Union

import numpy as np
from scipy.integrate import solve_ivp
import numba

# ---------------------------------------------------------------------------

# Numba‑compiled RHS function (fast for scalar ODEs)
@numba.njit  # supports Python 3.10
def _rober_routine(t: float, y: np.ndarray, k1: float, k2: float, k3: float) -> np.ndarray:
    """
    Computes f(t, y) for the Robertson system.
    Parameters are unrolled to avoid indexing overhead.
    """
    y1, y2, y3 = y[0], y[1], y[2]
    f0 = -k1 * y1 + k3 * y2 * y3
    f1 = k1 * y1 - k2 * y2 * y2 - k3 * y2 * y3
    f2 = k2 * y2 * y2
    return np.array([f0, f1, f2], dtype=np.float64)


# ---------------------------------------------------------------------------

class Solver:
    """
    Fast solver for the Robertson stiff ODE problem.
    """

    def solve(
        self,
        problem: Dict[str, Union[np.ndarray, float]],
    ) -> Dict[str, List[float]]:  # noqa: D401
        """
        Solve the Robertson ODE and return final state as a list.
        """
        sol = self._solve(problem, debug=False)
        if sol.success:
            return {
                "y1": [sol.y[0, -1]],
                "y2": [sol.y[1, -1]],
                "y3": [sol.y[2, -1]],
            }
        raise RuntimeError(f"Solver failed: {sol.message}")

    # -----------------------------------------------------------------------

    def _solve(
        self,
        problem: Dict[str, Union[np.ndarray, float]],
        debug: bool = True,
    ) -> Any:
        """
        Internal routine that performs the ODE integration.

        Parameters
        ----------
        problem:
            Dictionary containing:
                - 'y0': initial state (array-like of length 3)
                - 't0': start time
                - 't1': end time
                - 'k': tuple of three rate constants
        debug:
            If True, supply dense output and generate a dense time grid.
        """
        # Inject problem parameters into local variables
        y0 = np.asarray(problem["y0"], dtype=np.float64, order="C")
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        k1, k2, k3 = problem["k"]

        # Pack constants for the Numba routine
        def rhs(t, y):
            return _rober_routine(t, y, k1, k2, k3)

        # Integration settings
        rtol = 1e-11
        atol = 1e-9
        method = "Radau"

        if debug:
            # log‑spaced time grid for dense output
            t_eval = np.clip(
                np.exp(np.linspace(np.log(1e-6), np.log(t1), 1_000)),
                t0,
                t1,
            )
        else:
            t_eval = None

        sol = solve_ivp(
            rhs,
            (t0, t1),
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol