from typing import Any
import numpy as np
from scipy import signal

class Solver:

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[float]]:
        num = problem['num']
        den = problem['den']
        u = problem['u']
        t = problem['t']

        # Construct the state-space representation once
        system = signal.lti(num, den)

        # Directly simulate the response; this internally calls lfilter
        _, yout, _ = signal.lsim(system, u, t)

        return {"yout": yout.tolist()}