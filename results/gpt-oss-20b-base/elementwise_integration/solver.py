from typing import Any
import numpy as np
from scipy.integrate import quad
from scipy.special import wright_bessel

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        a = problem["a"]
        b = problem["b"]
        lower = problem["lower"]
        upper = problem["upper"]

        # Use a fast adaptive integrator without extra overhead of tanhsinh
        integ, _ = quad(
            lambda x: wright_bessel(a, b, x),
            lower,
            upper,
            limit=200,   # increase limit for stability
            points=[lower, upper],  # explicit points to avoid potential singularities
            epsabs=1e-12,
            epsrel=1e-12,
        )

        # Return a scalar or list compatible with the original implementation
        return {"result": [integ] if isinstance(integ, (np.ndarray, list)) else integ}