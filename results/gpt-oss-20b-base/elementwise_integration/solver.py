from typing import Any
import numpy as np
from scipy.integrate import quad
from scipy.special import wright_bessel

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Solve integration problem with scipy.integrate.quad."""
        a = problem['a']
        b = problem['b']
        lower = problem['lower']
        upper = problem['upper']

        # quad returns a tuple (value, error); we only need the value
        integral, _ = quad(wright_bessel, lower, upper, args=(a, b), limit=200)
        return {'result': float(integral)}