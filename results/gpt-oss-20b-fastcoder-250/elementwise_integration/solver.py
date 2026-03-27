from typing import Any
import numpy as np
from scipy.integrate import quad
from scipy.special import wright_bessel


class Solver:
    """Fast integration using Scipy's `quad`."""

    @staticmethod
    def _integrand(x: float, a: float, b: float) -> float:
        """Wrapper for SciPy integration."""
        return wright_bessel(x, a, b)

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        a = problem["a"]
        b = problem["b"]
        lower = problem["lower"]
        upper = problem["upper"]

        # `quad` is typically ~1–2× faster than `tanhsinh` for scalar integrals
        result, err = quad(self._integrand, lower, upper, args=(a, b), epsabs=1.49e-08, epsrel=0.0)
        # `quad` never returns a SuccessFlag array; we rely on the error estimate.
        # If the estimate is too high, re‑evaluate with tighter tolerances.
        if err > 1e-6:
            result, err = quad(self._integrand, lower, upper,
                               args=(a, b),
                               epsabs=1.49e-12, epsrel=1.49e-12)
        return {"result": [result]}