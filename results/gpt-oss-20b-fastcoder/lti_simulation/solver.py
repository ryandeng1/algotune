from typing import Any
import numpy as np
from scipy import signal

class Solver:
    """
    Solver for continuous‐time LTI systems using fast discrete‑time approximation.
    """

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[float]]:
        """
        Simulates the output of a continuous LTI system given the transfer function
        coefficients, the input signal and the time vector.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - 'num' : np.ndarray, numerator coefficients of H(s)
                - 'den' : np.ndarray, denominator coefficients of H(s)
                - 'u'   : np.ndarray, input signal values
                - 't'   : np.ndarray, time vector (assumed uniformly spaced)

        Returns
        -------
        dict
            Dictionary with key 'yout' mapping to a list of output signal samples.
        """
        num = problem["num"]
        den = problem["den"]
        u = problem["u"]
        t = problem["t"]

        # assume evenly spaced time steps; fallback to variable step if not
        if np.allclose(np.diff(t), t[1] - t[0]):
            dt = t[1] - t[0]
            # Convert continuous transfer function to discrete using bilinear transform
            b_disc, a_disc = signal.bilinear(num, den, dt)
            # Apply the filter to the input
            y = signal.lfilter(b_disc, a_disc, u)
        else:
            # Fallback to scipy's lsim for irregular time grids
            system = signal.lti(num, den)
            _, y, _ = signal.lsim(system, u, t)

        return {"yout": y.tolist()}