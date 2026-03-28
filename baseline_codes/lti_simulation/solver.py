from typing import Any
import numpy as np
from scipy import signal

class Solver:

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[float]]:
        """
        Solves the LTI simulation problem using scipy.signal.lsim.

        :param problem: A dictionary representing the LTI simulation problem.
        :return: A dictionary with key "yout" containing:
                 "yout": A list of floats representing the output signal.
        """
        num = problem['num']
        den = problem['den']
        u = problem['u']
        t = problem['t']
        system = signal.lti(num, den)
        tout, yout, xout = signal.lsim(system, u, t)
        solution = {'yout': yout.tolist()}
        return solution
