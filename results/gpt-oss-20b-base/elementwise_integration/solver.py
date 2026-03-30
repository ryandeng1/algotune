# solver.py
from typing import Any

import numpy as np
from scipy.integrate import tanhsinh
from scipy.special import wright_bessel


class Solver:
    """
    A lightweight solver that evaluates the integral

        ∫_{lower}^{upper} wright_bessel(t, a, b) dt

    using SciPy's tanhsinh quadrature.
    """

    # We expose the integrand as a staticmethod to allow potential external profiling
    # or replacement without changing the main solve logic.
    @staticmethod
    def _integrand(t: np.ndarray, a: float, b: float) -> np.ndarray:
        """Componentwise evaluation of the Wright Bessel integrand."""
        return wright_bessel(t, a, b)

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Compute the integral defined by 'lower', 'upper', 'a', and 'b'.

        Parameters
        ----------
        problem : dict
            Must contain the keys 'a', 'b', 'lower', and 'upper'.

        Returns
        -------
        dict
            Contains the key 'result', a Python list with the integral value.
        """
        a = problem["a"]
        b = problem["b"]
        lower = problem["lower"]
        upper = problem["upper"]

        # Use tanhsinh with a small tolerance to keep runtime short while keeping accuracy.
        res = tanhsinh(
            self._integrand,
            lower,
            upper,
            args=(a, b),
            epsabs=1e-12,
            epsrel=1e-10,
            maxiter=1000,
        )

        # The wrapper guarantees error handling; assert guarantees correctness.
        assert np.all(res.success)
        return {"result": res.integral.tolist()}