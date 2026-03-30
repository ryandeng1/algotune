# solver.py
from __future__ import annotations
from typing import Any, Dict
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    """
    Numerical solver for the Van der Pol oscillator

    The implementation is deliberately minimal: it simply forwards to
    :func:`scipy.integrate.solve_ivp`.  The only optimization is that
    argument construction is as cheap as possible and that the expensive
    `t_eval` array is only created when debugging is enabled.
    """

    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, list[float]]:
        """
        Solve the ODE and return the final state as a plain python list.
        """
        sol = self._solve(problem, debug=False)
        if sol.success:
            # Return the final value of each variable as a list of floats
            return dict(zip(["x", "v"], sol.y[:, -1].tolist()))
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: Dict[str, np.ndarray | float], debug: bool = True) -> Any:
        """
        Core integration routine.
        """
        # Fast construction of numpy arrays
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        mu = float(problem["mu"])

        # Localize variables for speed
        _mu = mu
        _rtol = 1e-8
        _atol = 1e-9
        _method = "Radau"

        def vdp(t, y):
            x, v = y
            return np.array([v, _mu * ((1 - x ** 2) * v - x)], dtype=np.float64)

        # Only request a dense output (and a fine t_eval) if debugging
        t_eval = np.linspace(t0, t1, 1000) if debug else None

        sol = solve_ivp(
            vdp,
            (t0, t1),
            y0,
            method=_method,
            rtol=_rtol,
            atol=_atol,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol