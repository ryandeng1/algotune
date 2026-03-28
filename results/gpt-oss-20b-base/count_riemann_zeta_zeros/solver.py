import math
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate the number of non-trivial zeta zeros with imaginary part
        <= t using the Riemann–von Mangoldt explicit formula
        N(t) = (t / (2π)) * log(t / (2π)) - t / (2π) + 7/8 + O(1/log t).
        The returned value is rounded to the nearest integer.
        """
        t = float(problem.get("t", 0))
        if t <= 0.0:
            return {"result": 0}

        two_pi = 2.0 * math.pi
        y = t / two_pi
        # Avoid log(0) for very small t
        if y <= 1e-10:
            return {"result": 0}

        n = y * math.log(y) - y + 7.0 / 8.0
        # Return integer count rounded to nearest integer
        return {"result": int(round(n))}