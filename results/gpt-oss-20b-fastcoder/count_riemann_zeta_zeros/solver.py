import math
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Count the number of non‑trivial zeros of the Riemann zeta function
        with imaginary part ≤ t.

        Uses the classical Riemann–von Mangoldt formula

            N(t) = (t / (2π)) * log(t / (2π)) - t / (2π) + 7/8 + O(1/t)

        which is accurate up to a small constant error for t > 20.
        For very small t we fall back to a hard‑coded table.
        """
        t = float(problem['t'])
        if t < 0:
            count = 0
        elif t < 2.5:
            # first zero at t ≈ 14.134..., so 0 for t < 2.5
            count = 0
        elif t < 20:
            # use small table of known zero heights
            known = [14.134725141, 21.022039638, 25.010857580, 30.424876125,
                     32.935061587, 37.586178159, 40.918719012, 43.327073281,
                     48.005150881, 49.773832477]
            count = sum(1 for h in known if h <= t)
        else:
            # Riemann–von Mangoldt approximation, rounded to nearest integer
            a = t / (2 * math.pi)
            count = a * (math.log(a) - 1) + 0.875
            count = int(round(count))
        return {'result': count}