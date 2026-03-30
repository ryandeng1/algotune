# solver.py
from __future__ import annotations
from typing import Dict, List
import numpy as np
from scipy import signal

class Solver:
    """
    LTI simulation solver.
    Uses `scipy.signal.lsim` to propagate a continuous‑time system
    defined by numerator `num` and denominator `den` coefficients.
    """

    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, List[float]]:
        """
        Solve the LTI simulation problem.

        Parameters
        ----------
        problem : dict
            ``problem`` must contain the keys ``'num'``, ``'den'``, ``'u'`` and ``'t'``.
            * ``num`` : 1‑dimensional array of numerator coefficients.
            * ``den`` : 1‑dimensional array of denominator coefficients.
            * ``u``   : array of input samples.
            * ``t``   : array of time stamps corresponding to ``u``.

        Returns
        -------
        dict
            ``{'yout': y}`` where ``y`` is a list of floating point numbers
            representing the system output at the supplied time points.
        """
        num: np.ndarray = problem["num"]
        den: np.ndarray = problem["den"]
        u: np.ndarray = problem["u"]
        t: np.ndarray = problem["t"]

        # Pre‑allocate the signal for speed
        system = signal.lti(num, den, dt=None)
        _, yout, _ = signal.lsim(system, u, t)

        # Convert the output to a Python list only once
        return {"yout": yout.tolist()}