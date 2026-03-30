from typing import Any, Dict
import numpy as np
from scipy.integrate import quad
from scipy.special import wright_bessel

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the definite integral of the Wright–Bessel function
        using SciPy's fast adaptive quadrature. The integration is
        performed over the range [lower, upper] with additional
        parameters a and b supplied to the integrand.

        Parameters
        ----------
        problem : dict
            Dictionary containing the keys 'a', 'b', 'lower', and
            'upper' that specify the integrand parameters and
            integration limits.

        Returns
        -------
        dict
            Dictionary with a single key 'result' containing the
            integral value as a Python scalar.
        """
        a = problem["a"]
        b = problem["b"]
        lower = problem["lower"]
        upper = problem["upper"]

        # scipy.integrate.quad is a highly optimised C implementation
        # and is typically faster than the tanhsinh algorithm for
        # smooth integrands such as the Wright–Bessel function.
        integral_val, _ = quad(
            wright_bessel,
            lower,
            upper,
            args=(a, b),
            limit=200,      # increase limits for difficult integrands
            epsabs=1e-12,   # high accuracy
            epsrel=1e-12,
        )

        # Convert float to Python scalar / list to match expected format
        return {"result": float(integral_val)}