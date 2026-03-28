import numpy as np
from scipy.signal import signal

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[float]]:
        """
        Solves the LTI system using scipy.signal.lsim for maximum speed.

        Parameters
        ----------
        problem : dict[str, np.ndarray]
            Dictionary containing:
            - 'num' : array of numerator coefficients
            - 'den' : array of denominator coefficients
            - 'u'   : input signal array
            - 't'   : time array

        Returns
        -------
        dict
            Dictionary with key 'yout' containing the output signal as a list of floats.
        """
        # Pull data from problem dict
        num, den, u, t = problem['num'], problem['den'], problem['u'], problem['t']

        # Create lti system once
        system = signal.lti(num, den)

        # Simulate system
        # lsim returns tout, yout, xout. tout is the same as t if shapes match
        tout, yout, _ = signal.lsim(system, u, t)

        # Convert the output to a list of python floats (fast conversion)
        return {"yout": yout.tolist()}