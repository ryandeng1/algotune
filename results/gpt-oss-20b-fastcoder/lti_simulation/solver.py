# solver.py
import numpy as np
from scipy import signal

class Solver:
    """
    An optimized LTI simulation solver.
    """

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[float]]:
        """
        Solves the LTI simulation problem using SciPy's signal.lsim.

        Parameters
        ----------
        problem : dict[str, np.ndarray]
            Dictionary containing the following keys:
                'num' : ndarray
                    Numerator coefficients of the transfer function.
                'den' : ndarray
                    Denominator coefficients of the transfer function.
                'u'   : ndarray
                    Input stimulus.
                't'   : ndarray
                    Time vector.

        Returns
        -------
        dict[str, list[float]]
            Dictionary with a single key 'yout' holding the output
            waveform as a plain Python list.
        """

        # Extract problem data (numpy arrays are already contiguous)
        num, den, u, t = (
            problem['num'],
            problem['den'],
            problem['u'],
            problem['t'],
        )

        # Build the system and run the simulation
        system = signal.lti(num, den)
        _, yout, _ = signal.lsim(system, u, t)

        # Convert the output to a plain Python list
        return {"yout": yout.tolist()}