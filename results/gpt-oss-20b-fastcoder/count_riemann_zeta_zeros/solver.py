import math
from typing import Any

class Solver:
    # Constant used in the Riemann–von Mangoldt formula
    _DEG2RAD = 1 / (2 * math.pi)

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Returns an estimate of the number of non‑trivial zeros of the Riemann
        zeta function with imaginary part ≤ t.
        The estimate uses the Riemann–von Mangoldt formula:
            N(t) = (t/(2π)) * ln(t/(2π)) - t/(2π) + 7/8 + O(1/t)
        For performance we ignore the O(1/t) term and truncate to an integer.
        """
        t: float = float(problem["t"])
        if t < 2:  # trivial small values
            return {"result": 0}

        factor = self._DEG2RAD * t
        n = factor * math.log(t / (2 * math.pi)) - factor + 0.875
        return {"result": int(round(n))}