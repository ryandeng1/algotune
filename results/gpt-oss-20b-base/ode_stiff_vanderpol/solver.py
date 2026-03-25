# solver.py
from typing import Any, List, Dict
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    """Solver for the stiff Van der Pol oscillator."""

    def _vdp_deriv(self, t: float, y: np.ndarray, mu: float) -> np.ndarray:
        """
        RHS of the Van der Pol system.

        Parameters
        ----------
        t : float
            Current time (unused, required by `solve_ivp`).
        y : np.ndarray
            Current state vector, [x, v].
        mu : float
            Stiffness parameter.

        Returns
        -------
        np.ndarray
            Derivatives [dx/dt, dv/dt].
        """
        x, v = y
        return np.array([v, mu * ((1 - x * x) * v - x)], dtype=np.float64)

    def solve(self, problem: Dict[str, Any]) -> List[float]:
        """
        Solve the Van der Pol equation from t0 to t1.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
            - 'mu'   : float, stiffness parameter
            - 'y0'   : list or array of length 2, initial state [x0, v0]
            - 't0'   : float, initial time
            - 't1'   : float, final time

        Returns
        -------
        list
            State [x(t1), v(t1)] at the final time.
        """
        mu = float(problem["mu"])
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])

        # Solve using a stiff solver (Radau) with tight tolerances
        sol = solve_ivp(
            fun=lambda t, y: self._vdp_deriv(t, y, mu),
            t_span=[t0, t1],
            y0=y0,
            method="Radau",
            rtol=1e-8,
            atol=1e-9,
            vectorized=False,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        # Return final state as plain Python list
        return sol.y[:, -1].tolist()
