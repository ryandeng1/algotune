from typing import Any
import numpy as np
from scipy.integrate import quad
from scipy.special import wright_bessel


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve integration problem using scipy.integrate.quad.
        The integrator should converge under default tolerances.
        """
        a = problem["a"]
        b = problem["b"]
        lower = problem["lower"]
        upper = problem["upper"]

        # Define the integrand
        func = lambda x: wright_bessel(x, a, b)

        # Use quad with adaptive precision controls
        integral, _ = quad(
            func,
            lower,
            upper,
            limit=200,
            epsabs=1.49e-8,
            epsrel=1.49e-8,
        )

        # Ensure the result is a list (to mimic original behaviour)
        return {"result": [integral]}