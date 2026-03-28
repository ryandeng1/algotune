import numpy as np
from scipy.integrate import quad
from scipy.special import wright_bessel

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Return the integral of the Wright-Bessel function over
        the supplied interval using a fast one‑dimensional quadrature.
        """
        a = problem["a"]
        b = problem["b"]
        lower = problem["lower"]
        upper = problem["upper"]

        # Simple 1‑dimensional quadrature – fast and accurate enough
        integral, _ = quad(wright_bessel, lower, upper, args=(a, b), limit=200)

        return {"result": float(integral)}