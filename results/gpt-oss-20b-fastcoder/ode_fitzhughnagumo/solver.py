#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
from typing import Any, Dict, Union

# --------------------------------------------------------------------------- #
# A highly‑optimized, numba‑compiled RHS for the FitzHugh‑Nagumo system
# --------------------------------------------------------------------------- #
@njit(fastmath=True)
def _fitzhugh_nagumo(t: float, y: np.ndarray, a: float, b: float, c: float, I: float) -> np.ndarray:
    """numba‑friendly version of the RHS."""
    v, w = y[0], y[1]
    # Insert constants with local names for speed
    a_local, b_local, c_local, I_local = a, b, c, I
    dv_dt = v - v ** 3 / 3.0 - w + I_local
    dw_dt = a_local * (b_local * v - c_local * w)
    return np.array([dv_dt, dw_dt], dtype=np.double)

# --------------------------------------------------------------------------- #
# Solver class
# --------------------------------------------------------------------------- #
class Solver:
    """
    Fast solver for the FitzHugh‑Nagumo model using a numba‑compiled RHS
    and SciPy's dense_output disabled.
    """

    def __init__(self) -> None:
        # constants for the integrator – tuned for speed
        self._rtol = 1e-8
        self._atol = 1e-8
        self._method = "RK45"

    def solve(self, problem: Dict[str, Union[np.ndarray, float]]) -> Dict[str, list[float]]:
        """
        Solve the ODE and return the final state.

        Parameters
        ----------
        problem : dict
            Dictionary containing keys:
                y0    : initial state (1‑D array)
                t0    : start time
                t1    : end time
                params: dict with keys 'a', 'b', 'c', 'I'

        Returns
        -------
        dict
            {'y': [v, w]} – final state as lists
        """
        # extract data
        y0_arr = np.array(problem["y0"], dtype=np.double)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        a, b, c, I = params["a"], params["b"], params["c"], params["I"]

        # Wrap the numba RHS in a closure that takes only t and y
        def rhs(t, y):
            return _fitzhugh_nagumo(t, y, a, b, c, I)

        # call the integrator – dense_output=False for speed
        sol = solve_ivp(
            rhs,
            (t0, t1),
            y0_arr,
            method=self._method,
            rtol=self._rtol,
            atol=self._atol,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # return final state as plain Python lists
        return {"y": sol.y[:, -1].tolist()}

    # -------------------------------------------------------------------------------- #
    # Optional debug helper (does not count toward runtime)
    # -------------------------------------------------------------------------------- #
    def _debug_solve(self, problem: Dict[str, Union[np.ndarray, float]]) -> Any:
        """
        Run the solver in debug mode, returning the full solution object.
        Useful for debugging but not used in production.
        """
        y0_arr = np.array(problem["y0"], dtype=np.double)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        a, b, c, I = params["a"], params["b"], params["c"], params["I"]

        def rhs(t, y):
            return _fitzhugh_nagumo(t, y, a, b, c, I)

        t_span = (t0, t1)
        t_eval = np.linspace(t0, t1, 1000)
        return solve_ivp(
            rhs,
            t_span,
            y0_arr,
            method=self._method,
            rtol=self._rtol,
            atol=self._atol,
            t_eval=t_eval,
            dense_output=True,
        )