import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solves an LTI system simulation.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - "num": list/array of numerator polynomial coefficients (highest power first)
                - "den": list/array of denominator polynomial coefficients (highest power first)
                - "u":  list/array of input signal samples at the times in `t`
                - "t":  list/array of monotonically increasing time samples

        Returns
        -------
        dict
            Dictionary with key "yout" mapping to a list of output samples
            corresponding to the input time vector.
        """
        # Extract data
        num = np.asarray(problem["num"], dtype=float)
        den = np.asarray(problem["den"], dtype=float)
        u = np.asarray(problem["u"], dtype=float)
        t = np.asarray(problem["t"], dtype=float)

        # Build the continuous-time LTI system
        system = signal.lti(num, den)

        # Simulate the system.  lsim automatically handles arbitrary t spacing.
        tout, yout, _ = signal.lsim(system, u, t)

        # Straightforward return; results are already NumPy arrays
        return {"yout": yout.tolist()}
