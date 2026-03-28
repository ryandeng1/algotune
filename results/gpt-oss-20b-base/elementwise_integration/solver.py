from typing import Any
import numpy as np
from scipy.special import wright_bessel
from scipy.integrate import quad

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Solve integration problem with scipy.integrate.quad for speed."""
        a = problem['a']
        b = problem['b']
        lower = problem['lower']
        upper = problem['upper']
        # define the integrand
        integrand = lambda x: wright_bessel(x, a, b)
        # use quad which is specifically optimized for this kind of one‑dimensional
        # integration and gives a single float result
        integral, _ = quad(integrand, lower, upper, limit=200, epsabs=1e-10, epsrel=1e-10)
        return {'result': [integral]}