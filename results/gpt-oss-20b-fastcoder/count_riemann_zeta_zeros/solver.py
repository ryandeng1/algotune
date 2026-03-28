import math
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Fast approximation to the number of non‑trivial zeta zeros
        with imaginary part <= t, using the Riemann‑von Mangoldt formula."""
        t = float(problem["t"])
        if t <= 0.0:
            return {"result": 0}
        # Riemann–von Mangoldt formula for N(t)
        mu = t / (2.0 * math.pi)
        # Avoid log(0) for very small t
        if mu <= 1e-12:
            nzeros = 0
        else:
            nzeros = int(math.floor(mu * math.log(mu) - mu + 7.0 / 8.0))
        return {"result": nzeros}