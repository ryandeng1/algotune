import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[float]]:
        # Extract problem data
        num = problem['num']
        den = problem['den']
        u = problem['u']
        t = problem['t']

        # Discretise the continuous system using zero‑order hold
        dt = t[1] - t[0]
        system_discrete = signal.cont2discrete((num, den), dt, method='zoh')
        b, a, _ = system_discrete

        # Compute the discrete output using lfilter (vectorised)
        y = signal.lfilter(b, a, u)

        return {'yout': y.tolist()}