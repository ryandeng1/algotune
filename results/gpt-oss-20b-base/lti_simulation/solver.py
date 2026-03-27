import numpy as np
from scipy.signal import lti, lsim


def solve(problem: dict[str, np.ndarray]) -> dict[str, list[float]]:
    """
    Solves an LTI simulation problem using scipy's lsim function.
    """
    num = problem["num"]
    den = problem["den"]
    u = problem["u"]
    t = problem["t"]

    # Build the LTI system and compute the response
    system = lti(num, den)
    tout, yout, _ = lsim(system, u, t)

    # Return the output as a plain Python list
    return {"yout": yout.tolist()}