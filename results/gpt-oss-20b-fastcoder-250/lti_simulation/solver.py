import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Simulate the continuous‑time LTI system defined by the transfer
        function H(s) = num(s) / den(s) for the given input signal u
        and time vector t.  The simulation is performed using
        scipy.signal.lsim, which integrates the state‑space model
        corresponding to the transfer function.

        Parameters
        ----------
        problem : dict
            Dictionary containing keys:
            - "num" : list[float]  numerator coefficients (highest power first)
            - "den" : list[float]  denominator coefficients (highest power first)
            - "u"   : list[float]  input signal values
            - "t"   : list[float]  monotonically increasing time points

        Returns
        -------
        dict
            Dictionary with key "yout" containing the simulated output
            signal as a list of floats.
        """
        # Extract problem data
        num = problem["num"]
        den = problem["den"]
        u = problem["u"]
        t = problem["t"]

        # Build the LTI system
        system = signal.lti(num, den)

        # Run the simulation
        tout, yout, _ = signal.lsim(system, u, t)

        # Ensure output length matches input time vector
        if len(tout) != len(t):
            # In rare cases lsim may produce different timestamps; align by interpolation
            yout = np.interp(t, tout, yout)

        return {"yout": yout.tolist()}
