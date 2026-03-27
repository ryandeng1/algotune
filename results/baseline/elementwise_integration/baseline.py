from typing import Any
import numpy as np
from scipy.integrate import tanhsinh
from scipy.special import wright_bessel


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Solve integration problem with scipy.integrate.tanhsinh."""
        a = problem["a"]
        b = problem["b"]
        lower = problem["lower"]
        upper = problem["upper"]
        res = tanhsinh(wright_bessel, lower, upper, args=(a, b))
        # This assert is a sanity check. The goal was to create problems where
        # the integrator always converges using the default tolerances. If this
        # ever fails, then `generate_problem` should be fixed.
        assert np.all(res.success)
        return {"result": res.integral.tolist()}
