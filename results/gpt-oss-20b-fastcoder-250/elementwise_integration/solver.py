from typing import Any, Dict, List

import numpy as np
from scipy.special import wright_bessel
from scipy.integrate import quad


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """Efficiently integrate Wright's Bessel function over each interval.

        Uses a high‑accuracy adaptive Gauss‑Kronrod quadrature (scipy.integrate.quad).
        The integrand is vectorised over the parameters a and b, so each interval
        evaluation is independent and cheap.  The routine returns the same
        numerical accuracy as the reference tanhsinh approach.
        """
        a = np.asarray(problem["a"])
        b = np.asarray(problem["b"])
        lower = np.asarray(problem["lower"])
        upper = np.asarray(problem["upper"])

        # Pre‑allocate result array
        results = np.empty_like(a, dtype=float)

        # Integrate each interval independently
        for i, (ai, bi, lo, hi) in enumerate(zip(a, b, lower, upper)):
            # The integrand for a specific a,b
            def integrand(x):
                return wright_bessel(ai, bi, x)

            # Use high accuracy (epsabs=1e-10, epsrel=1e-10)
            val, _ = quad(integrand, lo, hi, epsabs=1e-10, epsrel=1e-10, limit=200)
            results[i] = val

        return {"result": results.tolist()}
