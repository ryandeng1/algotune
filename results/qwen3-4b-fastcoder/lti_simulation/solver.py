from typing import Any
import numpy as np
from scipy import signal


class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[float]]:
        num = problem["num"]
        den = problem["den"]
        u = problem["u"]
        t = problem["t"]
        system = signal.lti(num, den)
        _, yout, _ = signal.lsim(system, u, t)
        return {"yout": yout.tolist()}