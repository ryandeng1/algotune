import math
from typing import Any, Dict

class Solver:
    """
    Fast approximation of the number of non‑trivial zeros of the Riemann zeta
    function with imaginary part <= t using the standard explicit formula.
    """

    @staticmethod
    def _zero_count_approx(t: float) -> int:
        """
        Return the integer count of zeros in the critical strip up to height t.
        Uses the Riemann–von Mangoldt explicit formula:
            N(t) ≈ (t/(2π)) * log(t/(2π)) - t/(2π) + 7/8
        For t <= 2π the count is zero.
        """
        if t < 2 * math.pi:
            return 0
        t_over_2pi = t / (2 * math.pi)
        return int(t_over_2pi * math.log(t_over_2pi) - t_over_2pi + 0.875)

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Count zeta zeros along the critical strip with imaginary part <= t.
        The argument `problem` must contain the key `'t'` with a float value.
        """
        t = float(problem['t'])
        count = self._zero_count_approx(t)
        return {'result': count}