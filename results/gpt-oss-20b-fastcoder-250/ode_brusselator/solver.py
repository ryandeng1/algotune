# solver.py
"""
Highly optimised Brusselator ODE solver.

This module defines a Solver class that forwards to a lightweight
SciPy ODE solver.  The implementation uses `scipy.integrate.solve_ivp`
with the 8th‑order Dormand-Prince (DOP853) method which provides
excellent accuracy with very few function evaluations for the
Brusselator equations.  The tolerance settings are chosen to match
the reference implementation while providing a small safety margin
to avoid failures on the edge cases.

No external dependencies other than NumPy and SciPy are required.
"""

from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    """
    Solver for the Brusselator model.

    The `solve` method expects a problem dictionary with the fields:

    - t0: float          initial time
    - t1: float          final time
    - y0: List[float]    initial concentrations [X0, Y0]
    - params: dict       {'A': float, 'B': float}

    It returns a NumPy array with the concentrations [X, Y] at `t1`.
    """

    # Class level constants – these are computed once during import.
    #: ODE solver tolerances (relative and absolute)
    _RTOL: float = 1e-8
    _ATOL: float = 1e-8
    #: Integration method to be used
    _METHOD: str = "DOP853"  # 8th‑order Dormand–Prince, performant for smooth ODEs

    def solve(self, problem: Dict[str, Any]) -> np.ndarray:
        """
        Solve the Brusselator equations for the given problem dictionary.

        Parameters
        ----------
        problem : dict
            Dictionary containing `t0`, `t1`, `y0`, and `params`.

        Returns
        -------
        np.ndarray
            Array of shape (2,) containing the concentrations
            [X(t1), Y(t1)].
        """
        # Extract problem data
        t0, t1 = problem["t0"], problem["t1"]
        y0 = np.asarray(problem["y0"], dtype=float)
        A = problem["params"]["A"]
        B = problem["params"]["B"]

        # Define the RHS once using a local closure that captures A,B
        def rhs(t: float, y: np.ndarray) -> np.ndarray:  # pragma: no cover
            X, Y = y
            return np.array([A + X**2 * Y - (B + 1) * X,
                             B * X - X**2 * Y],
                            dtype=float)

        # Perform integration
        sol = solve_ivp(
            rhs,
            [t0, t1],
            y0,
            method=self._METHOD,
            rtol=self._RTOL,
            atol=self._ATOL,
            dense_output=False,
            t_eval=None,
        )

        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")

        return sol.y[:, -1]
