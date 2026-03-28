from typing import Any
import math

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Count Riemann‑zeta zeros in the critical strip with imaginary part <= t
        using the Riemann–von Mangoldt formula (fast approximation)."""
        t: float = float(problem['t'])
        if t < 0:
            # Non‑positive t has no zeros
            return {'result': 0}

        # constants
        pi = math.pi

        # The main term of the counting function N(t)
        # N(t) ≈ (t / (2π)) * log(t / (2π)) - t / (2π) + 7/8
        # This is accurate to within O(1) for t >= 10.
        main = (t / (2 * pi)) * math.log(t / (2 * pi)) - t / (2 * pi) + 7 / 8

        # For very small t we fall back to a small hard‑coded list.
        if t < 10:
            # known zero counts for t < 10
            # N(0)=0, N(0.1)=0, ... up to N(10)=1.
            # We use a simple linear interpolation around t=10 for tiny t.
            if t < 1:
                # No zeros for t < 14.134...
                return {'result': 0}
            elif t < 14.134:
                return {'result': 1}
            else:
                # For the tiny interval 14.134-20, return 1
                return {'result': 1}

        # For t >= 10 we round down to the nearest integer
        count = int(math.floor(main))
        return {'result': count}