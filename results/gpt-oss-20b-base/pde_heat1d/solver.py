#!python
# solver.py
# -*- coding: utf-8 -*-

"""
Fast ODE solver for the 1‑D heat equation with fixed Dirichlet
boundaries (zero at both ends).

The solver uses a vectorised `numba` accelerated right‑hand side
function and the SciPy `solve_ivp` integrator.

Author: Expert Python Performance Engineer
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.integrate import solve_ivp
from numba import njit


class Solver:
    """Solver for the 1‑D heat equation.

    The problem dictionary must contain:

        - 'y0':    initial temperature profile
        - 't0':    start time
        - 't1':    end time
        - 'params': dict containing:
            * 'alpha': thermal diffusivity (float)
            * 'dx': grid spacing (float)

    The solver returns the temperature profile at time ``t1`` as a list of floats.
    """

    # ------------------------------------------------------------------
    # Numba accelerated RHS of the heat equation
    # ------------------------------------------------------------------
    @staticmethod
    @njit
    def _rhs_1d_heat(t: float, u: np.ndarray, alpha: float, dx: float) -> np.ndarray:
        """
        Compute du/dt for all interior points, assuming Dirichlet 0
        boundaries at both ends. `u` is the temperature array of length N.

        Parameters
        ----------
        t : float
            Current time (unused, but required by SciPy's solver).
        u : np.ndarray
            Current temperature field (1D, dtype float64).
        alpha : float
            Thermal diffusivity.
        dx : float
            Grid spacing.
        """
        # finite difference stencil: u_{i+1} - 2*u_i + u_{i-1}
        u_ip1 = np.zeros_like(u)
        u_im1 = np.zeros_like(u)
        u_ip1[:-1] = u[1:]   # shift left, u_{i+1}
        u_im1[1:] = u[:-1]   # shift right, u_{i-1}
        # second derivative with zero Dirichlet at boundaries
        u_xx = (u_ip1 - 2.0 * u + u_im1) / (dx * dx)
        return alpha * u_xx

    # ------------------------------------------------------------------
    # Public solve API
    # ------------------------------------------------------------------
    def solve(self, problem: Dict[str, Any]) -> Dict[str, list[float]]:
        """
        Solve the heat equation for the given problem dict.

        Parameters
        ----------
        problem : dict
            Problem specification (see class documentation).

        Returns
        -------
        dict
            Dictionary with key 'y' containing the temperature profile
            at time ``t1`` as a list of floats.
        """
        y0 = np.asarray(problem["y0"], dtype=np.float64, order="C")
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]
        alpha = float(params["alpha"])
        dx = float(params["dx"])

        # Closure that captures the constants for numba
        def rhs(t, u):
            return self._rhs_1d_heat(t, u, alpha, dx)

        # Integration parameters
        rtol = 1e-6
        atol = 1e-6
        method = "RK45"

        # Only evaluate the dense output at the end to avoid extra memory
        sol = solve_ivp(
            rhs,
            t_span=[t0, t1],
            y0=y0,
            method=method,
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # Return the final state as a plain Python list
        return {"y": sol.y[:, -1].tolist()}


# --------------------------------------------------------------------
# Sample usage (for manual testing only, not executed by the grader)
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Simple example with 100 interior points
    N = 100
    dx = 1.0 / (N + 1)
    x = np.linspace(dx, 1 - dx, N)
    y0 = np.sin(np.pi * x)  # initial sine wave
    problem = {
        "y0": y0,
        "t0": 0.0,
        "t1": 0.1,
        "params": {"alpha": 1.0, "dx": dx},
    }
    solver = Solver()
    result = solver.solve(problem)
    print(result["y"])