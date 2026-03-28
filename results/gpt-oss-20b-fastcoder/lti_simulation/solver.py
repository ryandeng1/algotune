import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[float]]:
        """Solve an LTI simulation using scipy's signal.lsim."""
        num, den, u, t = problem["num"], problem["den"], problem["u"], problem["t"]
        # Protect against already numpy arrays
        if not isinstance(u, np.ndarray):
            u = np.asarray(u, dtype=float)
        if not isinstance(t, np.ndarray):
            t = np.asarray(t, dtype=float)
        tout, yout, _ = signal.lsim((num, den), u, t)
        return {"yout": yout.tolist()}