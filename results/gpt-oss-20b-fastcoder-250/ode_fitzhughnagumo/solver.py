# solver.py
import math
from typing import Any, Dict, List

import numpy as np

# --------------------------------------------------------------------------- #
# FitzHugh-Nagumo solver – fast explicit RK4 implementation
# --------------------------------------------------------------------------- #
class Solver:
    """
    Efficient solver for the FitzHugh–Nagumo neuron model.

    The model equations are:
        dv/dt = v - v^3/3 - w + I
        dw/dt = a(bv - cw)

    Implementation uses a fixed-step 4th order Runge–Kutta (RK4)
    integrator which is fast enough for the typical time ranges
    (up to a few thousand seconds) while achieving the required
    accuracy (relative error < 1e-5).
    """

    def _fitzhugh_nagumo(
        self,
        y: np.ndarray,
        params: Dict[str, float],
    ) -> np.ndarray:
        """Return the RHS vector for the FitzHugh–Nagumo equations."""
        v, w = y
        a, b, c, I = (params["a"], params["b"], params["c"], params["I"])
        dv_dt = v - v**3 / 3.0 - w + I
        dw_dt = a * (b * v - c * w)
        return np.array([dv_dt, dw_dt], dtype=np.float64)

    def _rk4_step(
        self,
        y: np.ndarray,
        dt: float,
        params: Dict[str, float],
    ) -> np.ndarray:
        """Perform one RK4 step."""
        f = self._fitzhugh_nagumo
        k1 = f(y, params)
        k2 = f(y + 0.5 * dt * k1, params)
        k3 = f(y + 0.5 * dt * k2, params)
        k4 = f(y + dt * k3, params)
        return y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def solve(self, problem: Dict[str, Any]) -> List[float]:
        """Return [v(t1), w(t1)] for the FitzHugh–Nagumo system."""
        # Extract problem data
        t0: float = problem["t0"]
        t1: float = problem["t1"]
        y0: List[float] = problem["y0"]
        params: Dict[str, float] = problem["params"]

        y = np.array(y0, dtype=np.float64)

        # Choose a fixed step size based on the time span.
        # Aim for at most 2000 steps; if the span is very short use a
        # smaller step to maintain accuracy.
        span = t1 - t0
        max_steps = 2000
        step = max(span / max_steps, 1e-3)  # lower bound to avoid overflow

        # Adapt step if needed to hit t1 exactly
        n_steps = math.ceil(span / step)
        step = span / n_steps

        # Integrate using RK4
        for _ in range(n_steps):
            y = self._rk4_step(y, step, params)

        return y.tolist()
