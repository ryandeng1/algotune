import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, List[float]]:
        """
        A minimal, self‑contained ODE solver that works on the given problem
        dictionary.  The problem is expected to contain the following keys:

        * ``y0``: initial state (1‑D numpy array)
        * ``t_span``: tuple (t0, t1) defining the integration interval
        * ``t_eval``: optional detailed time grid for intermediate values
        * ``fun``: a callable ``fun(t, y) -> numpy.ndarray`` that returns the
          derivative of the state

        The method integrates the system using the classic 4‑th order
        Runge–Kutta (RK4) scheme.  For consistency with the original API,
        the result is returned as a dictionary containing a single list entry
        under the key ``"y"``.  The list contains the final state vector.
        """
        # Pull mandatory components out of the problem dict
        y0 = np.asarray(problem["y0"], dtype=float)
        t0, t1 = problem["t_span"]
        fun = problem["fun"]
        dt = 1e-3  # fixed step size – fine for quick tests

        # Create time points
        n_steps = int(np.ceil(abs(t1 - t0) / dt))
        t = t0 + np.arange(n_steps + 1) * dt
        if t[-1] > t1:
            t[-1] = t1

        # Initialize state array
        y = np.empty((len(t), y0.size), dtype=float)
        y[0] = y0

        # Simple RK4 integration
        for i in range(n_steps):
            ti = t[i]
            yi = y[i]
            k1 = fun(ti, yi)
            k2 = fun(ti + 0.5 * dt, yi + 0.5 * dt * k1)
            k3 = fun(ti + 0.5 * dt, yi + 0.5 * dt * k2)
            k4 = fun(ti + dt, yi + dt * k3)
            y[i + 1] = yi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Prepare output to match the original interface
        final_state = y[-1].tolist()
        return {"y": [final_state]}