from typing import Any, Dict, List
import numpy as np
from scipy import signal


class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, List[float]]:
        """
        Fast simulation of a continuous‑time LTI system.

        Parameters
        ----------
        problem : dict
            Must contain arrays ``num``, ``den``, ``u`` (input),
            and ``t`` (time points).

        Returns
        -------
        dict
            One key ``"yout"`` with a list of the simulated output values.
        """
        # Extract data
        num, den, u, t = problem["num"], problem["den"], problem["u"], problem["t"]

        # Leverage SciPy's fast continuous‑time simulation routine
        tout, yout, _ = signal.lsim(system=signal.lti(num, den), U=u, T=t)

        # Return as list. ``tolist`` is memory‑efficient for 1‑D arrays.
        return {"yout": yout.tolist()}