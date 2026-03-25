# solver.py

from typing import Any, List
import numpy as np
from scipy.integrate import solve_ivp


class Solver:
    """
    Fast solver for the Robertson chemical kinetics system.
    Uses SciPy's solve_ivp with the Radau method, suitable for stiff ODEs.
    """

    def solve(self, problem: dict[str, Any]) -> List[float]:
        """
        Solve the Robertson system from t0 to t1.

        Parameters
        ----------
        problem : dict
            Dictionary containing keys:
                - 't0' : initial time (float)
                - 't1' : final time (float)
                - 'y0' : initial concentrations [y1, y2, y3] (list of 3 floats)
                - 'k'  : rate constants [k1, k2, k3] (list of 3 floats)

        Returns
        -------
        list[float]
            Final concentrations [y1(t1), y2(t1), y3(t1)].
        """
        # Extract inputs
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        k1, k2, k3 = map(float, problem["k"])

        def rober(t, y):
            y1, y2, y3 = y
            f0 = -k1 * y1 + k3 * y2 * y3
            f1 = k1 * y1 - k2 * y2 ** 2 - k3 * y2 * y3
            f2 = k2 * y2 ** 2
            return np.array([f0, f1, f2], dtype=np.float64)

        # Use Radau for stiff problems; no intermediate evaluation to keep speed
        sol = solve_ivp(
            rober, (t0, t1), y0,
            method="Radau",
            rtol=1e-11,
            atol=1e-9,
            t_eval=None,      # only final time needed
            vectorized=False
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        return sol.y[:, -1].tolist()
