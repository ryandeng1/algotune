import numpy as np
from scipy import signal
from numba import njit

@njit
def solve_lti(num, den, u, t):
    num = np.array(num)
    den = np.array(den)
    u = np.array(u)
    t = np.array(t)
    
    system = signal.lti(num, den)
    tout, yout, _ = signal.lsim(system, u, t)
    return yout.tolist()

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[float]]:
        num = problem["num"]
        den = problem["den"]
        u = problem["u"]
        t = problem["t"]
        return {"yout": solve_lti(num, den, u, t)}
