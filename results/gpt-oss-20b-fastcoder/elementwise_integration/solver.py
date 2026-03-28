from typing import Any
import numpy as np
from scipy.integrate import tanhsinh
from scipy.special import wright_bessel

def solve(problem: dict[str, Any]) -> dict[str, Any]:
    """Fast integration using Scipy's tanhsinh quadrature."""
    a, b = problem['a'], problem['b']
    lower, upper = problem['lower'], problem['upper']

    # Perform the integral
    res = tanhsinh(wright_bessel, lower, upper, args=(a, b))
    if not np.all(res.success):
        raise RuntimeError("Integration failed")

    return {'result': res.integral.tolist()}