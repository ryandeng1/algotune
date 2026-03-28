from typing import Any
import numpy as np
from scipy.integrate import tanhsinh
from scipy.special import wright_bessel

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Integrate wright_bessel from lower to upper using tanh-sinh quadrature."""
        a = problem['a']
        b = problem['b']
        lower = problem['lower']
        upper = problem['upper']
        res = tanhsinh(wright_bessel, lower, upper, args=(a, b))
        # Ensure success; the API guarantees `res.success` is a boolean array
        assert np.all(res.success)
        return {'result': res.integral.tolist()}