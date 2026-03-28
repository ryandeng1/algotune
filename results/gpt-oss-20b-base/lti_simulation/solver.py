import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[float]]:
        """Fast LTI simulation using scipy.signal.lsim."""
        num, den, u, t = problem["num"], problem["den"], problem["u"], problem["t"]
        system = signal.lti(num, den)
        _, yout, _ = signal.lsim(system, u, t)
        return {"yout": yout.tolist()}