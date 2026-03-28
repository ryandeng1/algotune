from typing import Any
import numpy as np
from scipy import signal

class Solver:
    """
    Optimised LTI simulation solver for scipy 1.11+ (Python 3.10).
    """
    __slots__ = ("_num", "_den", "_t", "_u")

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[float]]:
        """
        Compute the output of a continuous‐time LTI system defined by a
        transfer function using ``scipy.signal.lsim``.
        """
        # Extract data directly from the dictionary for speed
        self._num, self._den, self._u, self._t = (
            problem["num"],
            problem["den"],
            problem["u"],
            problem["t"],
        )

        # Build the linear system once
        system = signal.lti(self._num, self._den)

        # Run the simulation
        _tout, _yout, _xout = signal.lsim(system, self._u, self._t)

        # Convert to a plain list of floats; this is faster than list(map(float, ...))
        return {"yout": _yout.tolist()}